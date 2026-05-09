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


def get_paper_trading_report() -> str:
    """One-call helper for the /paper-perf endpoint."""
    return format_paper_trading_html(compute_paper_trading_summary())
