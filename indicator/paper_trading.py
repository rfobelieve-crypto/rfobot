"""
Paper-trading (Stage 1 of indicator → quant migration).

NOT execution.  NOT live trading.  Reads tracked_signals (filled=1) and
computes the virtual PnL we would have realised if we had blindly opened
a 4h-hold position at every Strong/Moderate signal — taker fee +
slippage + funding included.

Why this lives here (not in research/):
    /paper-perf endpoint exposes this so we can monitor virtual PnL
    weekly without manual scripts.  This is the live observability layer
    we need before any real-money stage; once Strong-tier paper PnL is
    robustly positive over a forward window, we can graduate to the next
    stage (testnet → tiny live size).

Cost model (round-trip, in bps):
    taker fee : 10  (Binance perp 5 bps × 2 sides)
    slippage  :  2  (1 bp per side at typical sizes)
    funding   :  1  (4h hold ≈ half of 8h funding cycle, avg ~10bp/yr)
    TOTAL     : 13  ← every signal must average > +13 bps endpoint to be net-positive
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from shared.db import get_db_conn

logger = logging.getLogger(__name__)

COST_BPS = 13.0
COST = COST_BPS / 10000.0

# ── LDC swing dashboard constants ────────────────────────────────────────────
# Live cohort cutoff: 2026-05-13 16:00 UTC — when C.1 signal-aligned exit
# (cross AND ML signal flipped) replaced raw dynamic_cross exit.  Trades
# before this used the old exit logic and are excluded from "部署後實戰".
LDC_LIVE_CUTOFF = datetime(2026, 5, 13, 16, 0, 0)  # naive UTC, matches DB

# Backtest reference — 5/13 in-sample sweep of exit variants on 7-month
# BTC 1h history with v9 entry filter on.  Sample size is small (n=11)
# because v9 P>=0.55 is restrictive; numbers are directional, not
# statistically significant. Forward Stage 1 paper validation required.
# See research/dual_model/ldc_exit_optimization.py for the full sweep.
LDC_BACKTEST_BASELINE = {
    "n": 11,
    "wr_pct": 45.5,
    "pf": 1.42,
    "cum_1x_pct": 5.6,
    "cum_3x_pct": 16.8,
    "mdd_1x_pct": -4.5,
    "mdd_3x_pct": -13.5,
    "ratio": 1.24,
    "period_months": 7.0,
    "config": "v9 entry 0.55 + C.1 signal-cross exit",
    "note": "in-sample sweep, small n",
}

# Mirror of risk_manager defaults (risk_manager not wired in yet; alerts only).
LDC_KILL_SWITCH_MDD_1X_PCT = -20.0
LDC_KILL_SWITCH_WARN_PCT = -15.0    # warn at 75% of kill switch
LDC_CONSEC_LOSS_THRESHOLD = 7
LDC_STAGE2_MIN_TRADES = 100
LDC_STAGE2_NET_BPS = 5.0
LDC_MIN_HOLD_BARS = 4


def fetch_paper_signals(since: datetime | None = None) -> pd.DataFrame:
    """Pull settled tracked_signals into a DataFrame.

    `since` filters by signal_time >= cutoff (UTC datetime). None = all.
    """
    sql = """
        SELECT signal_time, direction, strength, confidence,
               entry_price, exit_price, actual_return_4h, regime,
               mag_pct_200, model_version
        FROM tracked_signals
        WHERE filled = 1
          AND entry_price IS NOT NULL
          AND exit_price  IS NOT NULL
          AND actual_return_4h IS NOT NULL
    """
    params: tuple = ()
    if since is not None:
        sql += " AND signal_time >= %s"
        params = (since.strftime("%Y-%m-%d %H:%M:%S"),)
    sql += " ORDER BY signal_time ASC"

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("entry_price", "exit_price", "actual_return_4h", "confidence"):
        df[col] = df[col].astype(float)
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    return df


def annotate_paper_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed endpoint PnL + net (after-cost) PnL columns."""
    if df.empty:
        return df
    sign = np.where(df["direction"] == "UP", 1.0, -1.0)
    df = df.copy()
    df["endpoint_ret"] = (df["exit_price"] / df["entry_price"] - 1.0) * sign
    df["twap_ret"] = df["actual_return_4h"] * sign
    df["endpoint_net"] = df["endpoint_ret"] - COST
    df["twap_net"] = df["twap_ret"] - COST
    df["win_endpoint"] = (df["endpoint_ret"] > 0).astype(int)
    return df


def _slice_metrics(group: pd.DataFrame) -> dict:
    n = len(group)
    if n == 0:
        return {"n": 0}
    e_net = group["endpoint_net"].values
    cum = np.cumsum(e_net)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    wins = e_net[e_net > 0]
    losses = e_net[e_net < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf
    return {
        "n": n,
        "wr": float(group["win_endpoint"].mean()),
        "avg_endpoint_bps": float(group["endpoint_ret"].mean()) * 10000,
        "avg_net_bps": float(e_net.mean()) * 10000,
        "sharpe_per_trade": (
            float(e_net.mean() / e_net.std()) if e_net.std() > 0 else 0.0
        ),
        "cum_net_pct": float(cum[-1]) * 100,
        "mdd_pct": float(drawdown.min()) * 100,
        "profit_factor": float(pf),
    }


def compute_paper_trading_summary(
    days_recent: int = 30,
) -> dict:
    """Compute the paper-trading dashboard.

    Returns a dict with: 'overall', 'recent', 'by_tier', 'by_conf_q' plus
    a 'cost_bps' field for transparency.  Empty if no signals.
    """
    df = annotate_paper_pnl(fetch_paper_signals())
    if df.empty:
        return {"empty": True, "cost_bps": COST_BPS}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_recent)
    recent_df = df[df["signal_time"] >= cutoff.replace(tzinfo=None)]

    by_tier: dict[str, dict] = {}
    for tier in ("Strong", "Moderate"):
        sub = df[df["strength"] == tier]
        if len(sub) == 0:
            continue
        by_tier[tier] = {
            "overall": _slice_metrics(sub),
            "by_dir": {
                d: _slice_metrics(sub[sub["direction"] == d])
                for d in ("UP", "DOWN") if len(sub[sub["direction"] == d]) > 0
            },
        }

    by_conf_q: dict[str, dict] = {}
    for tier in ("Strong", "Moderate"):
        sub = df[df["strength"] == tier].copy()
        if len(sub) < 50:
            continue
        try:
            sub["conf_q"] = pd.qcut(
                sub["confidence"], q=5,
                labels=["Q1_lo", "Q2", "Q3", "Q4", "Q5_hi"],
                duplicates="drop",
            )
        except ValueError:
            continue
        by_conf_q[tier] = {
            str(q): _slice_metrics(ssq)
            for q, ssq in sub.groupby("conf_q", observed=True)
            if len(ssq) >= 5
        }

    return {
        "empty": False,
        "cost_bps": COST_BPS,
        "n_total": len(df),
        "date_range": [
            df["signal_time"].min().isoformat(),
            df["signal_time"].max().isoformat(),
        ],
        "overall": _slice_metrics(df),
        "recent": _slice_metrics(recent_df) if len(recent_df) > 0 else {"n": 0},
        "recent_window_days": days_recent,
        "by_tier": by_tier,
        "by_conf_q": by_conf_q,
    }


def _fmt_bps(v: float, plus: bool = True) -> str:
    return f"{v:+.1f}" if plus else f"{v:.1f}"


def _fmt_metric_line(label: str, m: dict) -> str:
    if m.get("n", 0) == 0:
        return f"  {label} (n=0)"
    return (
        f"  {label} n={m['n']} "
        f"WR={m['wr']*100:.1f}% "
        f"net={_fmt_bps(m['avg_net_bps'])}bps "
        f"PF={m['profit_factor']:.2f}"
    )


def format_paper_trading_html(summary: dict) -> str:
    """Format compute_paper_trading_summary output for Telegram (HTML)."""
    if summary.get("empty"):
        return "📋 <b>Paper Trading</b>\n\n暫無已結算訊號可評估。"

    lines = ["📋 <b>Paper Trading (虛擬 PnL)</b>\n"]
    lines.append(
        f"成本模型: {summary['cost_bps']:.0f} bps round-trip "
        f"(taker 10 + slippage 2 + funding 1)\n"
    )

    o = summary["overall"]
    lines.append(
        f"<b>整體</b> n={o['n']} | "
        f"WR={o['wr']*100:.1f}% | "
        f"net={_fmt_bps(o['avg_net_bps'])} bps/trade | "
        f"PF={o['profit_factor']:.2f}\n"
        f"  累積 {_fmt_bps(o['cum_net_pct'])}% (cumsum) | MDD {o['mdd_pct']:.1f}%"
    )

    r = summary["recent"]
    lines.append(
        f"\n<b>最近 {summary['recent_window_days']} 天</b> "
        f"n={r.get('n', 0)} | "
        + (f"WR={r['wr']*100:.1f}% net={_fmt_bps(r['avg_net_bps'])} bps"
           if r.get("n", 0) > 0 else "(無樣本)")
    )

    for tier, t in summary["by_tier"].items():
        lines.append(f"\n<b>{tier}</b>")
        lines.append(_fmt_metric_line("整體", t["overall"]))
        for d, m in t["by_dir"].items():
            arrow = "🟢" if d == "UP" else "🔴"
            lines.append(_fmt_metric_line(f"{arrow} {d}", m))

    for tier, qs in summary["by_conf_q"].items():
        lines.append(f"\n<b>{tier} / 信心五分位</b>")
        for q in ["Q1_lo", "Q2", "Q3", "Q4", "Q5_hi"]:
            if q in qs:
                lines.append(_fmt_metric_line(q, qs[q]))

    lines.append(
        "\n<i>注意: 虛擬 PnL，未實際下單。樣本含多次模型重訓 + 5/9 confidence "
        "fix，前後段 distribution 不同；參考時請看「最近 X 天」切片。</i>"
    )
    return "\n".join(lines)


def fetch_hybrid_signals(since: datetime | None = None) -> pd.DataFrame:
    """Pull hybrid_signals (v9+LDC must-agree) with outcomes filled in.

    Returns empty DataFrame if table doesn't exist (e.g., before first
    hybrid signal has been emitted in production)."""
    sql = """
        SELECT signal_time, direction, p_long_win, p_short_win,
               ldc_signal, entry_price, exit_price, exit_reason,
               bars_held, gross_pct, net_pct_maker, win,
               model_version, paused_at_signal
        FROM hybrid_signals
        WHERE exit_price IS NOT NULL
    """
    params: tuple = ()
    if since is not None:
        sql += " AND signal_time >= %s"
        params = (since.strftime("%Y-%m-%d %H:%M:%S"),)
    sql += " ORDER BY signal_time ASC"

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return pd.DataFrame()
                raise
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("entry_price", "exit_price", "p_long_win", "p_short_win",
               "gross_pct", "net_pct_maker"):
        df[c] = df[c].astype(float)
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    df["win"] = df["win"].astype(int)
    df["paused_at_signal"] = df["paused_at_signal"].astype(int)
    return df


def _slice_hybrid_metrics(group: pd.DataFrame) -> dict:
    """Hybrid uses TP/SL exit + maker cost (5 bps) — net_pct_maker is already
    the net-of-cost return."""
    n = len(group)
    if n == 0:
        return {"n": 0}
    net = group["net_pct_maker"].values
    cum = np.cumsum(net)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    wins = net[net > 0]
    losses = net[net < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf
    return {
        "n": n,
        "wr": float((group["win"] == 1).mean()),
        "avg_net_bps": float(net.mean()) * 10000,
        "avg_gross_bps": float(group["gross_pct"].mean()) * 10000,
        "sharpe_per_trade": float(net.mean() / net.std()) if net.std() > 0 else 0.0,
        "cum_net_pct": float(cum[-1]) * 100,
        "mdd_pct": float(drawdown.min()) * 100,
        "profit_factor": float(pf),
    }


def compute_hybrid_summary(days_recent: int = 30) -> dict:
    """Compute hybrid cohort dashboard (v9+LDC must-agree)."""
    df = fetch_hybrid_signals()
    if df.empty:
        return {"empty": True, "cost_bps": 5.0, "label": "hybrid_v9_ldc"}

    # Only count signals that were not paused at emission
    df = df[df["paused_at_signal"] == 0]
    if df.empty:
        return {"empty": True, "cost_bps": 5.0, "label": "hybrid_v9_ldc",
                "note": "all signals paused by kill switch"}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_recent)
    recent = df[df["signal_time"] >= cutoff.replace(tzinfo=None)]

    by_dir: dict = {}
    for d in ("LONG", "SHORT"):
        sub = df[df["direction"] == d]
        if len(sub) > 0:
            by_dir[d] = _slice_hybrid_metrics(sub)

    return {
        "empty": False,
        "cost_bps": 5.0,
        "label": "hybrid_v9_ldc",
        "n_total": int(len(df)),
        "date_range": [
            df["signal_time"].min().isoformat(),
            df["signal_time"].max().isoformat(),
        ],
        "overall": _slice_hybrid_metrics(df),
        "recent": (_slice_hybrid_metrics(recent) if len(recent) > 0 else {"n": 0}),
        "recent_window_days": days_recent,
        "by_dir": by_dir,
    }


def format_hybrid_html(s: dict) -> str:
    """Format hybrid cohort summary for Telegram (HTML)."""
    if s.get("empty"):
        note = f" ({s['note']})" if s.get("note") else ""
        return f"\n\n🤝 <b>Hybrid v9+LDC</b>{note}\n暫無已結算 hybrid 訊號。"

    lines = ["\n\n🤝 <b>Hybrid v9+LDC must-agree</b>"]
    lines.append(
        f"成本: {s['cost_bps']:.0f} bps round-trip (maker only)\n"
        f"TP=0.5% / SL=0.3% / H=8h"
    )
    o = s["overall"]
    lines.append(
        f"\n<b>整體</b> n={o['n']} | WR={o['wr']*100:.1f}% | "
        f"net={o['avg_net_bps']:+.1f} bps/trade | PF={o['profit_factor']:.2f}\n"
        f"  累積 {o['cum_net_pct']:+.2f}% | MDD {o['mdd_pct']:.2f}%"
    )
    r = s["recent"]
    if r.get("n", 0) > 0:
        lines.append(
            f"\n<b>最近 {s['recent_window_days']} 天</b> n={r['n']} | "
            f"WR={r['wr']*100:.1f}% net={r['avg_net_bps']:+.1f} bps"
        )
    else:
        lines.append(f"\n<b>最近 {s['recent_window_days']} 天</b> (無樣本)")
    for d, m in s["by_dir"].items():
        lines.append(_fmt_metric_line(d, {**m, "avg_net_bps": m["avg_net_bps"]}))
    return "\n".join(lines)


def fetch_ldc_swing_positions(since: datetime | None = None) -> pd.DataFrame:
    """Pull ldc_swing_positions (closed only) with outcomes.

    Returns empty DataFrame if table doesn't exist (production not yet
    deployed) or no closed trades.
    """
    sql = """
        SELECT entry_time, exit_time, direction, entry_price, exit_price,
               exit_reason, bars_held, gross_pct, net_pct_maker, win,
               notional_usd, leverage, paused_at_signal, model_version
        FROM ldc_swing_positions
        WHERE status = 'CLOSED'
    """
    params: tuple = ()
    if since is not None:
        sql += " AND entry_time >= %s"
        params = (since.strftime("%Y-%m-%d %H:%M:%S"),)
    sql += " ORDER BY entry_time ASC"

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return pd.DataFrame()
                raise
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("entry_price", "exit_price", "gross_pct", "net_pct_maker",
               "notional_usd", "leverage"):
        df[c] = df[c].astype(float)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["win"] = df["win"].astype(int)
    df["paused_at_signal"] = df["paused_at_signal"].astype(int)
    df["bars_held"] = df["bars_held"].astype(int)
    return df


def _slice_swing_metrics(group: pd.DataFrame) -> dict:
    """LDC swing metrics — uses net_pct_maker (already cost-adjusted).

    Includes both peak-to-trough MDD (worst over history) and the CURRENT
    drawdown (cum - peak), the latter being what the kill-switch watches.
    """
    n = len(group)
    if n == 0:
        return {"n": 0}
    net = group["net_pct_maker"].values
    cum = np.cumsum(net)
    rmax = np.maximum.accumulate(cum)
    drawdown = cum - rmax
    wins = net[net > 0]
    losses = net[net < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf
    leverage = float(group["leverage"].iloc[0]) if "leverage" in group else 1.0
    avg_hold = float(group["bars_held"].mean())

    # Trailing loss streak (count consecutive losses from most-recent trade)
    g_sorted = group.sort_values("entry_time")
    streak = 0
    for w in g_sorted["win"].values[::-1]:
        if int(w) == 0:
            streak += 1
        else:
            break

    return {
        "n": n,
        "wr": float((group["win"] == 1).mean()),
        "avg_net_bps": float(net.mean()) * 10000,
        "avg_gross_bps": float(group["gross_pct"].mean()) * 10000,
        "cum_net_pct": float(cum[-1]) * 100,
        "cum_net_levered_pct": float(cum[-1]) * 100 * leverage,
        "mdd_pct": float(drawdown.min()) * 100,
        "mdd_levered_pct": float(drawdown.min()) * 100 * leverage,
        "current_dd_pct": float(drawdown[-1]) * 100,
        "current_dd_levered_pct": float(drawdown[-1]) * 100 * leverage,
        "profit_factor": float(pf),
        "leverage": leverage,
        "avg_hold_hours": avg_hold,
        "consec_loss_streak": int(streak),
    }


def _weekly_net_bps(df: pd.DataFrame, weeks: int = 4) -> list[dict]:
    """Bucket trades by ISO week, return last N weeks (most recent first)."""
    if df.empty:
        return []
    g = df.copy()
    g["week"] = g["entry_time"].dt.to_period("W-MON")
    out = []
    for wk, sub in g.groupby("week"):
        out.append({
            "week_start": sub["entry_time"].min().strftime("%m-%d"),
            "n": int(len(sub)),
            "avg_net_bps": float(sub["net_pct_maker"].mean()) * 10000,
            "wr": float((sub["win"] == 1).mean()),
        })
    out.sort(key=lambda r: r["week_start"], reverse=True)
    return out[:weeks]


def compute_ldc_swing_summary(
    days_recent: int = 30,
    since_cutoff: datetime | None = LDC_LIVE_CUTOFF,
) -> dict:
    """Compute LDC swing cohort dashboard.

    `since_cutoff` filters to the "deploy-after live" cohort (default =
    2026-05-13 commit when v9 filter + tuned LDC params went live).  Pass
    None to include every closed trade (informational only — params differ).
    """
    df = fetch_ldc_swing_positions(since=since_cutoff)
    if df.empty:
        return {"empty": True, "label": "ldc_swing", "cost_bps": 5.0,
                "cutoff": since_cutoff.isoformat() if since_cutoff else None}

    df = df[df["paused_at_signal"] == 0]
    if df.empty:
        return {"empty": True, "label": "ldc_swing", "cost_bps": 5.0,
                "note": "all signals paused",
                "cutoff": since_cutoff.isoformat() if since_cutoff else None}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_recent)
    recent = df[df["entry_time"] >= cutoff.replace(tzinfo=None)]

    by_dir: dict = {}
    for d in ("LONG", "SHORT"):
        sub = df[df["direction"] == d]
        if len(sub) > 0:
            by_dir[d] = _slice_swing_metrics(sub)

    return {
        "empty": False, "label": "ldc_swing", "cost_bps": 5.0,
        "cutoff": since_cutoff.isoformat() if since_cutoff else None,
        "n_total": int(len(df)),
        "date_range": [
            df["entry_time"].min().isoformat(),
            df["entry_time"].max().isoformat(),
        ],
        "overall": _slice_swing_metrics(df),
        "recent": _slice_swing_metrics(recent) if len(recent) > 0 else {"n": 0},
        "recent_window_days": days_recent,
        "by_dir": by_dir,
        "weekly_4w": _weekly_net_bps(df, weeks=4),
    }


# ── Dashboard helpers (mirror /perf layout) ──────────────────────────────────

def _fetch_recent_closed_ldc(limit: int = 5) -> pd.DataFrame:
    """Last N closed LDC swing trades (any params), most recent first."""
    sql = """
        SELECT entry_time, exit_time, direction, entry_price, exit_price,
               exit_reason, bars_held, gross_pct, net_pct_maker, win, leverage
        FROM ldc_swing_positions
        WHERE status = 'CLOSED'
        ORDER BY entry_time DESC
        LIMIT %s
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, (int(limit),))
                rows = cur.fetchall()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return pd.DataFrame()
                raise
    finally:
        conn.close()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("entry_price", "exit_price", "gross_pct", "net_pct_maker", "leverage"):
        df[c] = df[c].astype(float)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["win"] = df["win"].astype(int)
    df["bars_held"] = df["bars_held"].astype(int)
    return df


def _fetch_open_ldc_with_pnl() -> dict | None:
    """Current open LDC position enriched with live PnL.

    Returns None if no open position or the table is missing.
    """
    try:
        from indicator.ldc_swing_executor import get_open_position
    except Exception:
        return None
    try:
        pos = get_open_position()
    except Exception as exc:
        logger.warning("get_open_position failed: %s", exc)
        return None
    if not pos:
        return None

    # Latest BTC close from indicator_history (already 1h aligned).
    last_close = None
    last_bar_ts = None
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT `close`, dt FROM indicator_history "
                    "ORDER BY dt DESC LIMIT 1"
                )
                row = cur.fetchone()
                if row:
                    last_close = float(row["close"])
                    last_bar_ts = row["dt"]
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("latest close query failed: %s", exc)

    entry_ts = pos["entry_time"]
    if hasattr(entry_ts, "tz_convert"):
        try:
            entry_ts = entry_ts.tz_convert("UTC").tz_localize(None)
        except Exception:
            pass
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    bars_held = max(0, int((now_utc - entry_ts).total_seconds() / 3600))
    min_hold_remaining = max(0, LDC_MIN_HOLD_BARS - bars_held)

    entry_price = float(pos["entry_price"])
    direction = pos["direction"]
    leverage = float(pos.get("leverage") or 1.0)
    if last_close is not None:
        sign = 1.0 if direction == "LONG" else -1.0
        gross_pct = (last_close / entry_price - 1.0) * sign
        # Maker fee already paid 1 side; the other half hits on exit. Show
        # gross as the unrealised PnL — net only known at close.
        live_pnl_pct = gross_pct * 100
        live_pnl_levered_pct = live_pnl_pct * leverage
    else:
        live_pnl_pct = None
        live_pnl_levered_pct = None

    return {
        "id": int(pos["id"]),
        "direction": direction,
        "entry_time": entry_ts,
        "entry_price": entry_price,
        "leverage": leverage,
        "last_close": last_close,
        "last_bar_ts": last_bar_ts,
        "bars_held": bars_held,
        "min_hold_remaining": min_hold_remaining,
        "live_pnl_pct": live_pnl_pct,
        "live_pnl_levered_pct": live_pnl_levered_pct,
    }


def _compute_ldc_alerts(summary: dict) -> list[str]:
    """Generate alert lines for the LDC dashboard.

    Mirrors risk_manager thresholds (max_dd 20%, consec_loss 7), plus the
    Stage 2 graduation criterion (4 weeks net > +5 bps).
    """
    alerts: list[str] = []
    if summary.get("empty"):
        return alerts
    o = summary["overall"]

    # 1) Drawdown approaching kill switch (1x scale; 3x is implicit).
    cur_dd = o.get("current_dd_pct", 0.0)
    if cur_dd <= LDC_KILL_SWITCH_MDD_1X_PCT:
        alerts.append(
            f"🚨 1x 當前 DD {cur_dd:.1f}% 已達 kill switch "
            f"{LDC_KILL_SWITCH_MDD_1X_PCT:.0f}% — 應暫停或降階段"
        )
    elif cur_dd <= LDC_KILL_SWITCH_WARN_PCT:
        alerts.append(
            f"⚠️ 1x 當前 DD {cur_dd:.1f}% 接近 kill switch "
            f"({LDC_KILL_SWITCH_MDD_1X_PCT:.0f}%)"
        )

    # 2) Consecutive losses ≥ 7.
    streak = o.get("consec_loss_streak", 0)
    if streak >= LDC_CONSEC_LOSS_THRESHOLD:
        alerts.append(
            f"🚨 連續虧損 {streak} 筆 (≥{LDC_CONSEC_LOSS_THRESHOLD}) — "
            f"risk_manager 預設會暫停 24h"
        )
    elif streak >= LDC_CONSEC_LOSS_THRESHOLD - 2:
        alerts.append(f"⚠️ 連續虧損 {streak} 筆，距暫停門檻還 "
                       f"{LDC_CONSEC_LOSS_THRESHOLD - streak} 筆")

    # 3) Recent (≈4w) net bps turned negative.
    r = summary.get("recent", {})
    if r.get("n", 0) >= 20:
        rn = r["avg_net_bps"]
        if rn < 0:
            alerts.append(
                f"🔴 近 {summary['recent_window_days']} 天 net "
                f"{rn:+.1f} bps < 0 — Stage 2 進階倒退 (門檻 +{LDC_STAGE2_NET_BPS:.0f} bps)"
            )
        elif rn < LDC_STAGE2_NET_BPS:
            alerts.append(
                f"🟡 近 {summary['recent_window_days']} 天 net {rn:+.1f} bps，"
                f"低於 Stage 2 門檻 +{LDC_STAGE2_NET_BPS:.0f} bps"
            )

    # 4) Any of last 4 weekly buckets < 0 (more granular than #3).
    weeks = summary.get("weekly_4w", [])
    if weeks:
        bad = [w for w in weeks if w["avg_net_bps"] < 0]
        if bad:
            alerts.append(
                f"⚠️ 最近 4 週中有 {len(bad)} 週 net < 0 "
                f"(連續 4 週正 EV 才能進階)"
            )

    return alerts


def _format_stage2_progress(summary: dict) -> str:
    """Stage 1 → Stage 2 progress: trade count + 4-week net positive."""
    if summary.get("empty"):
        n = 0
    else:
        n = summary["overall"].get("n", 0)
    remaining_trades = max(0, LDC_STAGE2_MIN_TRADES - n)

    weeks = summary.get("weekly_4w", []) if not summary.get("empty") else []
    weeks_ge_threshold = sum(
        1 for w in weeks if w["avg_net_bps"] >= LDC_STAGE2_NET_BPS
    )

    pct = min(100, int(n / LDC_STAGE2_MIN_TRADES * 100))
    bar_len = 20
    filled = int(bar_len * n / LDC_STAGE2_MIN_TRADES)
    bar = "█" * min(filled, bar_len) + "░" * max(0, bar_len - filled)

    lines = ["\n<b>🎯 Stage 2 進階進度</b>"]
    lines.append(
        f"  Trades: <code>{bar}</code> {n}/{LDC_STAGE2_MIN_TRADES} "
        f"({pct}%, 差 {remaining_trades} 筆)"
    )
    lines.append(
        f"  4 週連續 net ≥ +{LDC_STAGE2_NET_BPS:.0f} bps: "
        f"{weeks_ge_threshold}/4 週達標"
    )
    return "\n".join(lines)


def _format_open_position_section(op: dict | None) -> str:
    """Render current open LDC position block; empty string if no position."""
    if op is None:
        return "\n<b>📦 當前持倉</b>\n  (無)"
    arrow = "🟢▲ LONG" if op["direction"] == "LONG" else "🔴▼ SHORT"
    lev = op["leverage"]
    entry_str = op["entry_time"].strftime("%m-%d %H:%M UTC")
    lines = [f"\n<b>📦 當前持倉</b> {arrow} (id={op['id']})"]
    lines.append(
        f"  entry: {entry_str} @ ${op['entry_price']:,.0f} | {lev:.0f}x"
    )
    if op["last_close"] is not None:
        lines.append(
            f"  now: ${op['last_close']:,.0f} | "
            f"PnL {op['live_pnl_pct']:+.2f}% (1x) / "
            f"<b>{op['live_pnl_levered_pct']:+.2f}% ({lev:.0f}x)</b>"
        )
    else:
        lines.append("  now: (無最新價)")
    if op["min_hold_remaining"] > 0:
        lines.append(
            f"  bars held: {op['bars_held']} (min hold 還剩 "
            f"{op['min_hold_remaining']} 根，期間無 exit)"
        )
    else:
        lines.append(
            f"  bars held: {op['bars_held']} (過 min hold，等 LDC cross / reverse)"
        )
    return "\n".join(lines)


def _format_recent_closed_section(df: pd.DataFrame) -> str:
    """Render last-N closed trades block; empty string if none."""
    if df is None or df.empty:
        return ""
    lines = ["\n<b>最近平倉 (5 筆)</b>"]
    for _, r in df.iterrows():
        d = r["direction"]
        arrow = "🟢▲" if d == "LONG" else "🔴▼"
        ts = r["entry_time"].strftime("%m-%d %H:%M")
        entry = float(r["entry_price"])
        net = float(r["net_pct_maker"]) * 100
        mark = "✅" if int(r["win"]) == 1 else "❌"
        reason = (r["exit_reason"] or "").strip()
        held = int(r["bars_held"])
        lines.append(
            f"  {arrow}[{d[0]}] {ts} ${entry:,.0f} → "
            f"{net:+.2f}% {mark} ({reason}, {held}h)"
        )
    return "\n".join(lines)


def format_ldc_swing_html(s: dict) -> str:
    """Format LDC swing summary for Telegram HTML, mirroring /perf layout."""
    header = ["\n\n📈 <b>LDC Swing (jdehorty + min hold 4h)</b>"]
    header.append(
        f"成本: {s.get('cost_bps', 5.0):.0f} bps round-trip (maker) | "
        f"無 TP/SL，C.1 signal-cross exit"
    )

    # ── Section 1: Backtest baseline (authoritative reference) ──
    b = LDC_BACKTEST_BASELINE
    header.append(
        f"\n<b>Backtest 基準</b> ({b.get('config', '')})"
    )
    header.append(
        f"  n={b['n']} | WR={b['wr_pct']:.1f}% | PF={b['pf']:.2f}"
    )
    header.append(
        f"  cum {b['cum_1x_pct']:+.1f}% (1x) / "
        f"<b>{b['cum_3x_pct']:+.1f}% (3x)</b>"
    )
    header.append(
        f"  MDD {b['mdd_1x_pct']:.1f}% (1x) / {b['mdd_3x_pct']:.1f}% (3x) "
        f"| cum/|MDD|={b['ratio']:.2f}"
    )
    if b.get("note"):
        header.append(f"  <i>({b['note']})</i>")

    # ── Section 2: Deploy-after live cohort ──
    lines = list(header)
    if s.get("empty"):
        note = f" ({s['note']})" if s.get("note") else ""
        lines.append(f"\n<b>部署後實戰</b> (2026-05-13~){note}")
        lines.append("  暫無已平倉訊號（v9 filter 開啟後尚無 trade 結算）")
    else:
        o = s["overall"]
        lev = o["leverage"]
        lines.append("\n<b>部署後實戰</b> (2026-05-13~ / v9 entry + C.1 exit)")
        lines.append(
            f"  n={o['n']} | WR={o['wr']*100:.1f}% | "
            f"net={o['avg_net_bps']:+.1f} bps/trade | PF={o['profit_factor']:.2f}"
        )
        lines.append(
            f"  cum {o['cum_net_pct']:+.2f}% (1x) / "
            f"<b>{o['cum_net_levered_pct']:+.2f}% ({lev:.0f}x)</b>"
        )
        lines.append(
            f"  MDD {o['mdd_pct']:.2f}% (1x) / "
            f"<b>{o['mdd_levered_pct']:.2f}% ({lev:.0f}x)</b> | "
            f"當前 DD {o['current_dd_pct']:.2f}%"
        )
        lines.append(f"  avg hold {o['avg_hold_hours']:.0f}h")

        # Recent window slice
        r = s["recent"]
        if r.get("n", 0) > 0:
            lines.append(
                f"\n<b>最近 {s['recent_window_days']} 天</b> n={r['n']} | "
                f"WR={r['wr']*100:.1f}% | net={r['avg_net_bps']:+.1f}bp | "
                f"cum {r['cum_net_levered_pct']:+.2f}% ({lev:.0f}x)"
            )
        else:
            lines.append(f"\n<b>最近 {s['recent_window_days']} 天</b> (無樣本)")

        # LONG / SHORT split
        for d, m in s.get("by_dir", {}).items():
            icon = "🟢" if d == "LONG" else "🔴"
            lines.append(
                f"  {icon} {d} n={m['n']} WR={m['wr']*100:.1f}% "
                f"net={m['avg_net_bps']:+.1f}bp cum={m['cum_net_pct']:+.2f}%"
            )

        # Weekly breakdown (4 most recent ISO weeks)
        weeks = s.get("weekly_4w", [])
        if weeks:
            lines.append("\n<b>週切片 (最近 4 週)</b>")
            for w in weeks:
                badge = "🟢" if w["avg_net_bps"] >= LDC_STAGE2_NET_BPS else (
                    "🟡" if w["avg_net_bps"] >= 0 else "🔴"
                )
                lines.append(
                    f"  {badge} {w['week_start']} n={w['n']} "
                    f"WR={w['wr']*100:.0f}% net={w['avg_net_bps']:+.1f}bp"
                )

    # ── Section 3: Current open position ──
    try:
        op = _fetch_open_ldc_with_pnl()
    except Exception as exc:
        logger.warning("open position fetch failed: %s", exc)
        op = None
    lines.append(_format_open_position_section(op))

    # ── Section 4: Recent 5 closed ──
    try:
        recent_df = _fetch_recent_closed_ldc(limit=5)
        rc_block = _format_recent_closed_section(recent_df)
        if rc_block:
            lines.append(rc_block)
    except Exception as exc:
        logger.warning("recent closed fetch failed: %s", exc)

    # ── Section 5: Alerts ──
    alerts = _compute_ldc_alerts(s)
    if alerts:
        lines.append("\n<b>⚠️ 警報</b>")
        lines.extend(f"  {a}" for a in alerts)

    # ── Section 6: Stage 2 progress ──
    lines.append(_format_stage2_progress(s))

    return "\n".join(lines)


# ── V7 paper cohort (v7.1 signal-exit + 3xATR trailing stop) ─────────────────
# Parallel Stage-1 paper cohort, runs alongside LDC. Equity-compounding with
# fixed-fractional 2%-risk / 1x sizing (see indicator/v7_paper_executor.py).

V7_INITIAL_CAPITAL = 1000.0

# Backtest reference — research/v71_v7_sizing_1x.py: WF-OOS 5-month one-position
# sim, fixed-fractional 2% risk + 1x cap. Live uses the production model and
# will differ — that is exactly what this Stage-1 cohort is validating.
V7_BACKTEST_BASELINE = {
    "n": 169, "wr_pct": 56.2, "roi_pct": 84.9, "mdd_pct": 5.3,
    "sharpe": 5.10, "avg_eq_ret_pct": 0.38,
    "config": "2% risk / 1x / 3×ATR trailing + signal exit",
    "note": "WF-OOS backtest — live uses production model, expect drift",
}


def fetch_v7_paper_positions(since: datetime | None = None) -> pd.DataFrame:
    """Pull v7_paper_positions (closed only). Empty df if table absent."""
    sql = """
        SELECT entry_time, exit_time, direction, entry_price, exit_price,
               entry_tier, exit_reason, bars_held, gross_pct, net_pct,
               equity_ret_pct, equity_before, equity_after, win,
               size_frac, notional_usd, model_version, paused_at_signal
        FROM v7_paper_positions
        WHERE status = 'CLOSED'
    """
    params: tuple = ()
    if since is not None:
        sql += " AND entry_time >= %s"
        params = (since.strftime("%Y-%m-%d %H:%M:%S"),)
    sql += " ORDER BY entry_time ASC"

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return pd.DataFrame()
                raise
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("entry_price", "exit_price", "gross_pct", "net_pct",
              "equity_ret_pct", "equity_before", "equity_after",
              "size_frac", "notional_usd"):
        df[c] = df[c].astype(float)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["win"] = df["win"].astype(int)
    df["bars_held"] = df["bars_held"].astype(int)
    df["paused_at_signal"] = df["paused_at_signal"].fillna(0).astype(int)
    return df


def _slice_v7_metrics(group: pd.DataFrame) -> dict:
    """V7 metrics — equity-compounding (fixed-fractional 2%-risk sizing)."""
    n = len(group)
    if n == 0:
        return {"n": 0}
    g = group.sort_values("exit_time")
    eq_ret = g["equity_ret_pct"].values            # percent, per trade
    eq_after = g["equity_after"].values
    start = float(g["equity_before"].iloc[0])
    curve = np.concatenate([[start], eq_after])    # equity path incl. seed
    peak = np.maximum.accumulate(curve)
    dd = (curve / peak - 1.0) * 100.0
    final = float(eq_after[-1])
    wins = eq_ret[eq_ret > 0]
    losses = eq_ret[eq_ret < 0]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else np.inf
    # trailing loss streak
    streak = 0
    for w in g["win"].values[::-1]:
        if int(w) == 0:
            streak += 1
        else:
            break
    return {
        "n": n,
        "wr": float((group["win"] == 1).mean()),
        "avg_eq_ret_pct": float(eq_ret.mean()),
        "med_eq_ret_pct": float(np.median(eq_ret)),
        "best_pct": float(eq_ret.max()),
        "worst_pct": float(eq_ret.min()),
        "avg_hold_h": float(group["bars_held"].mean()),
        "avg_net_bps": float(group["net_pct"].mean()) * 10000,
        "final_equity": final,
        "roi_pct": float((final / V7_INITIAL_CAPITAL - 1.0) * 100.0),
        "mdd_pct": float(dd.min()),
        "sharpe_per_trade": (
            float(eq_ret.mean() / eq_ret.std()) if eq_ret.std() > 0 else 0.0
        ),
        "profit_factor": float(pf),
        "consec_loss_streak": int(streak),
    }


def compute_v7_summary(days_recent: int = 30) -> dict:
    """Compute V7 paper cohort dashboard.

    Trades stamped paused_at_signal=1 (model-version transition shadow —
    see v7_paper_executor._is_in_transition_window) are excluded from all
    cohort stats: they ran on an unproven freshly-retrained model and must
    not contaminate the Stage-1 graduation decision.
    """
    df_all = fetch_v7_paper_positions()
    if df_all.empty:
        return {"empty": True, "label": "v7_paper"}

    n_shadow = int((df_all["paused_at_signal"] == 1).sum())
    df = df_all[df_all["paused_at_signal"] == 0].copy()
    if df.empty:
        return {"empty": True, "label": "v7_paper", "n_shadow": n_shadow,
                "note": "all V7 trades shadowed (model-transition window)"}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days_recent)
    recent = df[df["entry_time"] >= cutoff.replace(tzinfo=None)]

    by_dir: dict = {}
    for d in ("LONG", "SHORT"):
        sub = df[df["direction"] == d]
        if len(sub) > 0:
            by_dir[d] = _slice_v7_metrics(sub)

    by_reason: dict = {}
    for reason, sub in df.groupby("exit_reason"):
        by_reason[str(reason)] = {
            "n": int(len(sub)),
            "avg_eq_ret_pct": float(sub["equity_ret_pct"].mean()),
            "wr": float((sub["win"] == 1).mean()),
        }

    return {
        "empty": False, "label": "v7_paper",
        "n_total": int(len(df)),
        "n_shadow": n_shadow,
        "date_range": [
            df["entry_time"].min().isoformat(),
            df["entry_time"].max().isoformat(),
        ],
        "overall": _slice_v7_metrics(df),
        "recent": _slice_v7_metrics(recent) if len(recent) > 0 else {"n": 0},
        "recent_window_days": days_recent,
        "by_dir": by_dir,
        "by_reason": by_reason,
    }


def _fetch_open_v7_with_pnl() -> dict | None:
    """Current open V7 position with live unrealised PnL. None if flat."""
    try:
        from indicator.v7_paper_executor import get_open_position
    except Exception:
        return None
    try:
        pos = get_open_position()
    except Exception as exc:
        logger.warning("v7 get_open_position failed: %s", exc)
        return None
    if not pos:
        return None

    last_close = None
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT `close` FROM indicator_history "
                            "ORDER BY dt DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    last_close = float(row["close"])
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("v7 latest close query failed: %s", exc)

    entry_ts = pos["entry_time"]
    if hasattr(entry_ts, "tz_convert"):
        try:
            entry_ts = entry_ts.tz_convert("UTC").tz_localize(None)
        except Exception:
            pass
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    bars_held = max(0, int((now_utc - entry_ts).total_seconds() / 3600))

    entry_price = float(pos["entry_price"])
    direction = pos["direction"]
    live_pnl_pct = None
    if last_close is not None:
        sign = 1.0 if direction == "LONG" else -1.0
        live_pnl_pct = (last_close / entry_price - 1.0) * sign * 100.0

    return {
        "id": int(pos["id"]),
        "direction": direction,
        "entry_time": entry_ts,
        "entry_price": entry_price,
        "entry_tier": pos.get("entry_tier"),
        "current_stop": float(pos["current_stop"]),
        "size_frac": float(pos["size_frac"]),
        "last_close": last_close,
        "bars_held": bars_held,
        "live_pnl_pct": live_pnl_pct,
        "paused": int(pos.get("paused_at_signal", 0) or 0),
    }


def format_v7_html(s: dict) -> str:
    """Format V7 paper cohort summary for Telegram HTML."""
    lines = ["\n\n🤖 <b>V7 Paper (v7.1 訊號出場 + 3×ATR trailing)</b>"]
    lines.append(
        "本金 $1000 | 2% 風險 / 1x / 複利 | 成本 8 bps round-trip"
    )

    # Backtest reference
    b = V7_BACKTEST_BASELINE
    lines.append(f"\n<b>Backtest 基準</b> ({b['config']})")
    lines.append(
        f"  n={b['n']} | WR={b['wr_pct']:.1f}% | ROI {b['roi_pct']:+.1f}% | "
        f"MDD {b['mdd_pct']:.1f}% | Sharpe {b['sharpe']:.2f}"
    )
    lines.append(f"  <i>({b['note']})</i>")

    if s.get("empty"):
        if s.get("n_shadow"):
            lines.append(
                f"\n<b>實盤 paper</b>\n  {s['n_shadow']} 筆 V7 訊號全為 "
                f"shadow（模型轉換窗口）— 暫無計入 cohort 的樣本"
            )
        else:
            lines.append("\n<b>實盤 paper</b>\n  暫無已平倉 V7 訊號（尚未結算）")
    else:
        o = s["overall"]
        lines.append("\n<b>實盤 paper</b>")
        if s.get("n_shadow"):
            lines.append(
                f"  <i>（{s['n_shadow']} 筆 shadow 已排除 — 模型轉換窗口）</i>"
            )
        lines.append(
            f"  n={o['n']} | WR={o['wr']*100:.1f}% | "
            f"每筆 {o['avg_eq_ret_pct']:+.2f}% | PF={o['profit_factor']:.2f}"
        )
        lines.append(
            f"  權益 ${o['final_equity']:,.0f} | "
            f"<b>ROI {o['roi_pct']:+.1f}%</b> | MDD {o['mdd_pct']:.1f}%"
        )
        lines.append(
            f"  avg hold {o['avg_hold_h']:.0f}h | "
            f"Sharpe/trade {o['sharpe_per_trade']:.2f} | "
            f"最佳/最差 {o['best_pct']:+.1f}%/{o['worst_pct']:+.1f}%"
        )

        r = s["recent"]
        if r.get("n", 0) > 0:
            lines.append(
                f"\n<b>最近 {s['recent_window_days']} 天</b> n={r['n']} | "
                f"WR={r['wr']*100:.1f}% | 每筆 {r['avg_eq_ret_pct']:+.2f}%"
            )
        else:
            lines.append(f"\n<b>最近 {s['recent_window_days']} 天</b> (無樣本)")

        for d, m in s.get("by_dir", {}).items():
            icon = "🟢" if d == "LONG" else "🔴"
            lines.append(
                f"  {icon} {d} n={m['n']} WR={m['wr']*100:.1f}% "
                f"每筆 {m['avg_eq_ret_pct']:+.2f}%"
            )

        if s.get("by_reason"):
            lines.append("\n<b>出場原因</b>")
            for reason, m in s["by_reason"].items():
                lines.append(
                    f"  {reason}: n={m['n']} WR={m['wr']*100:.0f}% "
                    f"每筆 {m['avg_eq_ret_pct']:+.2f}%"
                )

    # Open position
    try:
        op = _fetch_open_v7_with_pnl()
    except Exception as exc:
        logger.warning("v7 open position fetch failed: %s", exc)
        op = None
    if op is None:
        lines.append("\n<b>📦 當前持倉</b>\n  (無)")
    else:
        arrow = "🟢▲ LONG" if op["direction"] == "LONG" else "🔴▼ SHORT"
        shadow = " <i>[SHADOW]</i>" if op.get("paused") else ""
        lines.append(f"\n<b>📦 當前持倉</b> {arrow} (id={op['id']}){shadow}")
        lines.append(
            f"  entry: {op['entry_time'].strftime('%m-%d %H:%M')} "
            f"@ ${op['entry_price']:,.0f} ({op.get('entry_tier') or '-'})"
        )
        if op["live_pnl_pct"] is not None:
            lines.append(
                f"  now: ${op['last_close']:,.0f} | "
                f"PnL {op['live_pnl_pct']:+.2f}% | "
                f"trailing stop ${op['current_stop']:,.0f} | "
                f"held {op['bars_held']}h"
            )
        else:
            lines.append(
                f"  trailing stop ${op['current_stop']:,.0f} | "
                f"held {op['bars_held']}h"
            )

    return "\n".join(lines)


def get_paper_trading_report() -> str:
    """One-call helper for the /paper-perf endpoint.

    Returns combined v7 indicator (tracked_signals) + LDC swing
    + V7 paper (v7_paper_positions) + legacy hybrid report.
    """
    v7_html = format_paper_trading_html(compute_paper_trading_summary())
    try:
        ldc_html = format_ldc_swing_html(compute_ldc_swing_summary())
    except Exception as exc:
        logger.warning("ldc_swing summary failed: %s", exc)
        ldc_html = "\n\n📈 <b>LDC Swing</b>\n(查詢失敗 — 表格可能尚未建立)"
    try:
        v7_paper_html = format_v7_html(compute_v7_summary())
    except Exception as exc:
        logger.warning("v7 paper summary failed: %s", exc)
        v7_paper_html = "\n\n🤖 <b>V7 Paper</b>\n(查詢失敗 — 表格可能尚未建立)"
    try:
        hybrid_summary = compute_hybrid_summary()
        if hybrid_summary.get("empty") and not hybrid_summary.get("note"):
            hybrid_html = ""  # don't show empty legacy section
        else:
            hybrid_html = format_hybrid_html(hybrid_summary)
    except Exception as exc:
        logger.warning("hybrid summary failed: %s", exc)
        hybrid_html = ""
    return v7_html + ldc_html + v7_paper_html + hybrid_html
