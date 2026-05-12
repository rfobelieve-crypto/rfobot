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
    """LDC swing metrics — uses net_pct_maker (already cost-adjusted)."""
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
    return {
        "n": n,
        "wr": float((group["win"] == 1).mean()),
        "avg_net_bps": float(net.mean()) * 10000,
        "avg_gross_bps": float(group["gross_pct"].mean()) * 10000,
        "cum_net_pct": float(cum[-1]) * 100,
        "cum_net_levered_pct": float(cum[-1]) * 100 * leverage,
        "mdd_pct": float(drawdown.min()) * 100,
        "mdd_levered_pct": float(drawdown.min()) * 100 * leverage,
        "profit_factor": float(pf),
        "leverage": leverage,
        "avg_hold_hours": avg_hold,
    }


def compute_ldc_swing_summary(days_recent: int = 30) -> dict:
    """Compute LDC swing cohort dashboard."""
    df = fetch_ldc_swing_positions()
    if df.empty:
        return {"empty": True, "label": "ldc_swing", "cost_bps": 5.0}

    df = df[df["paused_at_signal"] == 0]
    if df.empty:
        return {"empty": True, "label": "ldc_swing", "cost_bps": 5.0,
                "note": "all signals paused"}

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
        "n_total": int(len(df)),
        "date_range": [
            df["entry_time"].min().isoformat(),
            df["entry_time"].max().isoformat(),
        ],
        "overall": _slice_swing_metrics(df),
        "recent": _slice_swing_metrics(recent) if len(recent) > 0 else {"n": 0},
        "recent_window_days": days_recent,
        "by_dir": by_dir,
    }


def format_ldc_swing_html(s: dict) -> str:
    """Format LDC swing summary for Telegram HTML."""
    if s.get("empty"):
        note = f" ({s['note']})" if s.get("note") else ""
        return f"\n\n📈 <b>LDC Swing (jdehorty)</b>{note}\n暫無已平倉訊號。"

    lines = ["\n\n📈 <b>LDC Swing (jdehorty + min hold 4h)</b>"]
    lines.append(
        f"成本: {s['cost_bps']:.0f} bps round-trip (maker) | "
        f"無 TP/SL，dynamic_cross exit"
    )
    o = s["overall"]
    lev = o["leverage"]
    lines.append(
        f"\n<b>整體</b> n={o['n']} | WR={o['wr']*100:.1f}% | "
        f"net={o['avg_net_bps']:+.1f} bps/trade | PF={o['profit_factor']:.2f}\n"
        f"  累積 {o['cum_net_pct']:+.2f}% (1x) / "
        f"<b>{o['cum_net_levered_pct']:+.2f}% ({lev:.0f}x)</b>\n"
        f"  MDD {o['mdd_pct']:.2f}% (1x) / "
        f"<b>{o['mdd_levered_pct']:.2f}% ({lev:.0f}x)</b> | "
        f"avg hold {o['avg_hold_hours']:.0f}h"
    )
    r = s["recent"]
    if r.get("n", 0) > 0:
        lines.append(
            f"\n<b>最近 {s['recent_window_days']} 天</b> n={r['n']} | "
            f"WR={r['wr']*100:.1f}% | net={r['avg_net_bps']:+.1f}bp | "
            f"cum {r['cum_net_levered_pct']:+.2f}% ({lev:.0f}x)"
        )
    else:
        lines.append(f"\n<b>最近 {s['recent_window_days']} 天</b> (無樣本)")
    for d, m in s["by_dir"].items():
        lines.append(
            f"  {d} n={m['n']} WR={m['wr']*100:.1f}% "
            f"net={m['avg_net_bps']:+.1f}bp cum={m['cum_net_pct']:+.2f}%"
        )
    return "\n".join(lines)


def get_paper_trading_report() -> str:
    """One-call helper for the /paper-perf endpoint.

    Returns combined v7 (tracked_signals) + LDC swing (ldc_swing_positions)
    + legacy hybrid (hybrid_signals — kept for reference) report.
    """
    v7_html = format_paper_trading_html(compute_paper_trading_summary())
    try:
        ldc_html = format_ldc_swing_html(compute_ldc_swing_summary())
    except Exception as exc:
        logger.warning("ldc_swing summary failed: %s", exc)
        ldc_html = "\n\n📈 <b>LDC Swing</b>\n(查詢失敗 — 表格可能尚未建立)"
    try:
        hybrid_summary = compute_hybrid_summary()
        if hybrid_summary.get("empty") and not hybrid_summary.get("note"):
            hybrid_html = ""  # don't show empty legacy section
        else:
            hybrid_html = format_hybrid_html(hybrid_summary)
    except Exception as exc:
        logger.warning("hybrid summary failed: %s", exc)
        hybrid_html = ""
    return v7_html + ldc_html + hybrid_html
