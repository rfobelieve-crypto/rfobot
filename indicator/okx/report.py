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
from indicator.timeutil import fmt_tpe

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


# ── Gate B (Stage 3 → 4 promotion gate) ──────────────────────────────


# Minimum / target sample sizes for the trade-layer edge gate.
# Lower = sample-size pressure too soon, upper = practical promotion target.
GATE_B_SAMPLE_MIN = 30
GATE_B_SAMPLE_TARGET = 50


def bootstrap_mean_ci_bps(
    net_pcts: list[float],
    n_iter: int = 2000,
    seed: int = 42,
) -> Optional[tuple[float, float]]:
    """Percentile bootstrap 95% CI on mean net_pct, returned in bps.

    None if n < 5 (CI meaningless with tiny samples). Pure-stdlib (random)
    so no scipy import inflates dashboard latency.
    """
    n = len(net_pcts)
    if n < 5:
        return None
    import random
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [net_pcts[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_iter * 0.025)]
    hi = means[int(n_iter * 0.975)]
    return (lo * 10000, hi * 10000)


def compute_gate_b_status(
    n_closed: int,
    net_pcts: list[float],
    avg_net_bps: float,
) -> dict:
    """Where are we on Stage 3 → 4a promotion?

    Pass criteria (CLAUDE.md compressed Gate B):
      - n_closed ≥ GATE_B_SAMPLE_MIN (30)
      - avg_net_bps ≥ 0
      - bootstrap 95% CI lower bound > 0  (statistically reject "no edge")

    Returns dict with status (one of "accumulating", "passed", "failed",
    "marginal"), progress %, CI bounds, and a human-readable summary.
    """
    progress_pct = min(100.0, n_closed / GATE_B_SAMPLE_TARGET * 100)
    ci = bootstrap_mean_ci_bps(net_pcts) if n_closed >= 5 else None

    if n_closed < GATE_B_SAMPLE_MIN:
        status = "accumulating"
        summary = (f"累積中 {n_closed}/{GATE_B_SAMPLE_MIN} "
                   f"(目標 {GATE_B_SAMPLE_TARGET})")
    else:
        if avg_net_bps < 0:
            status = "failed"
            summary = f"avg net {avg_net_bps:+.1f} bps < 0 — edge 未驗到"
        elif ci is None or ci[0] <= 0:
            status = "marginal"
            ci_str = (f"95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]"
                      if ci else "CI 無法計算")
            summary = (f"avg net {avg_net_bps:+.1f} bps > 0 但 {ci_str} "
                       f"下緣未離 0")
        else:
            status = "passed"
            summary = (f"avg net {avg_net_bps:+.1f} bps, "
                       f"95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] — Gate B 通過")

    return {
        "status": status,
        "n_closed": n_closed,
        "sample_min": GATE_B_SAMPLE_MIN,
        "sample_target": GATE_B_SAMPLE_TARGET,
        "progress_pct": progress_pct,
        "avg_net_bps": avg_net_bps,
        "ci_lo_bps": ci[0] if ci else None,
        "ci_hi_bps": ci[1] if ci else None,
        "summary": summary,
    }


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
                           equity_before, notional_usd, size_contracts
                    FROM v7_okx_positions
                    WHERE status IN ('CLOSED', 'DEMOTED')
                      AND (model_version IS NULL
                           OR model_version NOT LIKE 'manual_test%')
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


INITIAL_CAPITAL_USD = 155.0   # original Stage 3 deposit (2026-05-31, $154.86)

# Live-P&L baseline. History: 2026-06-05 manual blow-up → $105.15 refund
# deposited 2026-06-07 → operator temporarily withdrew the whole balance
# while FLAT (temporary cash need, NOT a loss — equity → $0.01 tripped
# CAP-4 DEMOTE and broke the since-6/7 M2M curve), then re-deposited
# $197.55 (informed capital top-up, user decision 2026-07-14), then
# deposited further to $1218.44 (2026-07-24, 6th informed override — see
# CLAUDE.md §Stage 3 資本再加碼至 $1218.44), then a SECOND manual blow-up
# on 2026-07-27 (a 37.11-contract LONG the executor never opened; equity
# 1218 → $16.62 over ~10 hours) followed by a redeposit to $274.
#
# The executor placed no orders in that window — its last fill was id=20 on
# 2026-07-16 and it sat HALTed on CAP-2 throughout — so resetting the
# headline baseline here is not laundering strategy losses; it is excluding
# activity the strategy did not perform. Trade-count gates (Gate B / shadow)
# are NOT reset — they continue.
EXECUTOR_RESTART_CAPITAL_USD = 274.0
EXECUTOR_RESTART_SINCE = "2026-07-28"


def _get_equity_curve_stats(since: str) -> dict:
    """Peak / trough / max-DD / current-DD-from-peak of live M2M equity.

    Uses ALL balance snapshots (not daily close) so peak == the trailing-peak
    drawdown alert's peak (state.get_peak_equity = MAX since `since`). Keeps
    dashboard consistent with the Telegram M2M drawdown alert.
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT total_eq_usd FROM v7_okx_balance_snapshots "
                "WHERE ts >= %s ORDER BY ts", (since,))
            eqs = [float(r["total_eq_usd"]) for r in cur.fetchall()
                   if r.get("total_eq_usd") is not None]
    except Exception:
        logger.exception("equity_curve_stats_failed")
        return {}
    finally:
        conn.close()
    if len(eqs) < 2:
        return {}
    peak, mdd = eqs[0], 0.0
    for v in eqs:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak * 100)
    hi, cur_eq = max(eqs), eqs[-1]
    return {"peak_usd": hi, "trough_usd": min(eqs), "mdd_pct": mdd,
            "cur_dd_pct": (cur_eq / hi - 1.0) * 100.0 if hi else 0.0}


def _get_btc_benchmark(since: str) -> dict:
    """BTC buy-and-hold return + MDD over the same window (indicator_history)."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT close FROM indicator_history "
                "WHERE dt >= %s AND close IS NOT NULL ORDER BY dt", (since,))
            px = [float(r["close"]) for r in cur.fetchall()]
    except Exception:
        logger.exception("btc_benchmark_query_failed")
        return {}
    finally:
        conn.close()
    if len(px) < 2:
        return {}
    peak, mdd = px[0], 0.0
    for v in px:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak * 100)
    return {"btc_ret_pct": (px[-1] - px[0]) / px[0] * 100, "btc_mdd_pct": mdd}


def compute_okx_summary() -> dict:
    """Aggregate stats for the OKX live cohort.

    Returns a dict suitable for both JSON and templated rendering.
    """
    balance = _get_latest_balance()
    state = _get_executor_state()
    closed = _get_closed_trades()
    open_pos = _get_open_position()
    kills = _get_recent_kill_log(days=7)

    # Trade stats — three WR definitions reconciled side by side.
    # gross  : gross_pct > 0                    (price moved right way)
    # net    : net_pct   > 0                    (gross - 8 bps assumed cost)
    # equity : equity_after > equity_before     (wallet truth — real fees,
    #                                            funding, slippage all baked in)
    n = len(closed)
    wins_gross = sum(1 for t in closed if (t.get("gross_pct") or 0) > 0)
    wins_net = sum(1 for t in closed if (t.get("net_pct") or 0) > 0)
    wins_equity = sum(
        1 for t in closed
        if (t.get("equity_after") is not None
            and t.get("equity_before") is not None
            and t["equity_after"] > t["equity_before"])
    )
    avg_bps = (sum((t.get("net_pct") or 0) for t in closed) / n * 10000
               if n else 0.0)
    cum_pct = (sum((t.get("net_pct") or 0) for t in closed) * 100
               if n else 0.0)
    cum_equity_pct = (sum((t.get("equity_ret_pct") or 0) for t in closed)
                      if n else 0.0)

    # ── Professional metrics ────────────────────────────────────────────
    _nets = [float(t.get("net_pct") or 0) for t in closed]
    _wins = [x for x in _nets if x > 0]
    _losses = [x for x in _nets if x < 0]
    avg_win_pct = (sum(_wins) / len(_wins) * 100) if _wins else 0.0
    avg_loss_pct = (sum(_losses) / len(_losses) * 100) if _losses else 0.0
    payoff_ratio = (avg_win_pct / abs(avg_loss_pct)) if _losses else None
    profit_factor = (sum(_wins) / abs(sum(_losses))) if _losses else None
    cost_drag_bps = (sum(((t.get("gross_pct") or 0) - (t.get("net_pct") or 0))
                         for t in closed) / n * 10000) if n else 0.0
    cum_gross_pct = (sum((t.get("gross_pct") or 0) for t in closed) * 100
                     if n else 0.0)

    def _side_stats(side: str) -> dict:
        sub = [t for t in closed if (t.get("direction") or "").upper() == side]
        ns = len(sub)
        w = sum(1 for t in sub if (t.get("net_pct") or 0) > 0)
        return {"n": ns,
                "cum_pct": sum((t.get("net_pct") or 0) for t in sub) * 100,
                "wr": (w / ns * 100 if ns else 0.0)}
    long_stats, short_stats = _side_stats("LONG"), _side_stats("SHORT")
    eqc = _get_equity_curve_stats(EXECUTOR_RESTART_SINCE)
    mdd_pct = eqc.get("mdd_pct")
    benchmark = _get_btc_benchmark(EXECUTOR_RESTART_SINCE)

    # Sharpe — per-trade + naive annualised by observed cadence
    net_pcts = [float(t.get("net_pct") or 0) for t in closed]
    pt_sharpe = per_trade_sharpe(net_pcts)
    gate_b = compute_gate_b_status(n, net_pcts, avg_bps)
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
    base_cap = EXECUTOR_RESTART_CAPITAL_USD
    eq_pct_from_initial = (
        ((current_eq - base_cap) / base_cap * 100)
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
        "initial_capital_usd": base_cap,
        "capital_basis_since": EXECUTOR_RESTART_SINCE,
        "eq_pct_from_initial": eq_pct_from_initial,
        "executor_status": (state.get("status") if state else "UNKNOWN"),
        "executor_reason": (state.get("reason") if state else None),
        "executor_changed_at": (state.get("last_changed_at") if state else None),
        "n_closed": n,
        # Gross WR kept under legacy keys to preserve external consumers
        # (Telegram /okx-perf, dashboards). Net + equity WR added alongside.
        "wins": wins_gross,
        "win_rate_pct": (wins_gross / n * 100 if n else 0.0),
        "wins_net": wins_net,
        "win_rate_pct_net": (wins_net / n * 100 if n else 0.0),
        "wins_equity": wins_equity,
        "win_rate_pct_equity": (wins_equity / n * 100 if n else 0.0),
        "avg_net_bps": avg_bps,
        "cum_net_pct": cum_pct,
        "cum_equity_pct": cum_equity_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "cost_drag_bps": cost_drag_bps,
        "cum_gross_pct": cum_gross_pct,
        "long_stats": long_stats,
        "short_stats": short_stats,
        "mdd_pct": mdd_pct,
        "equity_peak_usd": eqc.get("peak_usd"),
        "equity_cur_dd_pct": eqc.get("cur_dd_pct"),
        "benchmark": benchmark,
        "sharpe_per_trade": pt_sharpe,
        "sharpe_annualised": ann_sharpe,
        "gate_b": gate_b,
        "open_position": open_pos,
        "recent_trades": closed[-5:],
        "kill_log_7d": kills,
    }


def format_okx_report(summary: dict) -> str:
    """Telegram-friendly HTML report."""
    lines: list[str] = []
    lines.append("<b>OKX LIVE Stage 3 · 2x 有效槓桿</b>")
    lines.append("")

    # Account
    eq = summary.get("current_equity_usd")
    if eq is not None:
        delta = summary.get("eq_pct_from_initial") or 0.0
        age = summary.get("balance_age_sec") or 0
        lines.append(
            f"💰 Equity: ${eq:.2f}  "
            f"({delta:+.2f}% since 補資 ${summary['initial_capital_usd']:.2f} "
            f"@ {summary.get('capital_basis_since', '')})")
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

    # Trade stats — show all three WR definitions so the gap between
    # "price moved right way" and "wallet actually grew" is transparent.
    n = summary.get("n_closed", 0)
    if n == 0:
        lines.append("📊 <i>No closed trades yet</i>")
    else:
        wr_gross = summary["win_rate_pct"]
        wr_net = summary.get("win_rate_pct_net", wr_gross)
        wr_eq = summary.get("win_rate_pct_equity", wr_gross)
        lines.append(f"📊 Trades: {n}")
        lines.append(
            f"   WR gross:  {summary['wins']}/{n} = {wr_gross:.0f}%  "
            f"<i>(price direction)</i>")
        lines.append(
            f"   WR net:    {summary.get('wins_net', summary['wins'])}/{n} = "
            f"{wr_net:.0f}%  <i>(after 8bps cost)</i>")
        lines.append(
            f"   WR equity: {summary.get('wins_equity', summary['wins'])}/{n} = "
            f"{wr_eq:.0f}%  <i>(wallet truth)</i>")
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

    # Gate B (Stage 3 → 4a promotion progress)
    gb = summary.get("gate_b")
    if gb:
        gate_icon = {
            "accumulating": "🟡",
            "marginal": "🟠",
            "passed": "🟢",
            "failed": "🔴",
        }.get(gb["status"], "⚪")
        lines.append(f"{gate_icon} Gate B (Stage 3→4a): {gb['summary']}")
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
                f"   #{t['id']} {fmt_tpe(t.get('entry_time'))} {t['direction']} "
                f"→ {t.get('exit_reason','')} {sign}{bps:.0f} bps")
        lines.append("")

    # Kill triggers
    kills = summary.get("kill_log_7d") or []
    if kills:
        lines.append(f"⚠️ Kill log (7d, {len(kills)} entries):")
        for k in kills[:3]:
            lines.append(f"   {fmt_tpe(k['ts'])} {k['trigger_id']} {k['severity']}")
    return "\n".join(lines)


def get_okx_report() -> str:
    """Public entry: compute + format."""
    try:
        summary = compute_okx_summary()
        return format_okx_report(summary)
    except Exception:
        logger.exception("get_okx_report_failed")
        return "❌ OKX report query failed"
