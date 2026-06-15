"""Dashboard Tab: Stability — 6 graduation milestones to Stage 4.

Rationale: V7 has edge (sign_AUC 0.59, top-5% WR 73%) but the bigger
risk now is operational/execution, not model accuracy.  This tab
tracks the concrete checkpoints to confirm Stage 3 is "stable enough"
before scaling up to Stage 4a ($1k).

Six milestones (per 2026-06-02 stability-first decision):
  M1. 30+ live trades closed
  M2. 90 days with 0 unresolved kill triggers
  M3. Monthly IC >= +0.10 for 3 consecutive months on production data
  M4. Railway uptime: 95%+ of expected hourly bars present
  M5. >=1 regime change observed during live operation
  M6. >=1 kill trigger fired AND recovered (not just unit tested)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta

from indicator.timeutil import fmt_tpe

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from indicator.dashboard_tabs._components import (
    card, section, status_badge, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_stability() -> str:
    """Top-level: 6 milestones grid + drift monitor + kill recovery log."""
    parts = [
        section("Stage 3 → 4 進階里程碑", "milestones", True,
                _build_milestones()),
        section("Kill Trigger Recovery History", "kills", True,
                _build_kill_recovery()),
        section("Operational Uptime (update_cycle coverage)", "uptime", True,
                _build_uptime()),
    ]
    return "\n".join(parts)


# ── M1-M6 milestones grid ────────────────────────────────────────────


def _milestone_card(title: str, current: float, target: float,
                     unit: str = "", subtitle: str = "") -> str:
    """Progress card — green if hit, yellow if partial, red if 0%."""
    pct = (current / target * 100) if target > 0 else 0
    pct = min(100, pct)
    if pct >= 100:
        color = "#36ffae"
        status = "PASS"
    elif pct >= 50:
        color = "#f5b544"
        status = f"{pct:.0f}%"
    else:
        color = "#ff5f6d"
        status = f"{pct:.0f}%"
    val_str = f"{current:.0f}{unit}" if current == int(current) \
                                       else f"{current:.1f}{unit}"
    return f"""
    <div style="background:rgba(154,160,166,0.03);border-left:3px solid {color};
                 padding:14px 16px;border-radius:6px;font-family:inherit">
      <div style="color:rgba(154,160,166,0.8);font-size:11px;
                  letter-spacing:0.05em;margin-bottom:4px">{title}</div>
      <div style="color:#FFFFFF;font-size:22px;font-weight:700;
                  margin-bottom:2px">{val_str}
        <span style="font-size:14px;color:{color};margin-left:8px">{status}</span>
      </div>
      <div style="color:rgba(154,160,166,0.8);font-size:11px">
        target {target}{unit}  ·  {subtitle}
      </div>
      <div style="background:rgba(255,255,255,0.05);height:4px;
                  border-radius:2px;margin-top:8px;overflow:hidden">
        <div style="background:{color};height:100%;
                    width:{pct:.0f}%;transition:width 0.3s"></div>
      </div>
    </div>"""


def _build_milestones() -> str:
    cards = []

    # M1: 30+ live trades closed
    try:
        n_live = _count_live_trades()
        cards.append(_milestone_card(
            "M1: Live Trades", n_live, 30, " trades",
            "OKX live closed positions"))
    except Exception as e:
        cards.append(f'<div style="color:#ff5f6d">M1 err: {e}</div>')

    # M2: 90 days 0 unresolved kill triggers
    try:
        days_clean = _days_since_last_unresolved_kill()
        cards.append(_milestone_card(
            "M2: Clean Days", days_clean, 90, "d",
            "days since last unresolved kill trigger"))
    except Exception as e:
        cards.append(f'<div style="color:#ff5f6d">M2 err: {e}</div>')

    # M3: Monthly IC >= 0.10 for 3 consecutive months
    try:
        n_consec = _consecutive_strong_ic_months(threshold=0.10)
        cards.append(_milestone_card(
            "M3: Strong-IC Months", n_consec, 3, " months",
            "consecutive months IC >= +0.10"))
    except Exception as e:
        cards.append(f'<div style="color:#ff5f6d">M3 err: {e}</div>')

    # M4: Railway 95%+ uptime (last 30 days of expected hourly bars)
    try:
        uptime_pct = _expected_bar_coverage_pct(days=30)
        cards.append(_milestone_card(
            "M4: Uptime (30d)", uptime_pct, 95, "%",
            "expected hourly bars actually written"))
    except Exception as e:
        cards.append(f'<div style="color:#ff5f6d">M4 err: {e}</div>')

    # M5: regime change observed
    try:
        n_regimes_seen = _distinct_regimes_in_live_window()
        cards.append(_milestone_card(
            "M5: Regime Diversity", n_regimes_seen, 2, "",
            "distinct regimes during live operation"))
    except Exception as e:
        cards.append(f'<div style="color:#ff5f6d">M5 err: {e}</div>')

    # M6: kill trigger fired + recovered
    try:
        recovery_count = _kill_recovery_count()
        cards.append(_milestone_card(
            "M6: Recovery Validated", recovery_count, 1, "",
            "kill trigger fired AND auto-resumed (real, not stress)"))
    except Exception as e:
        cards.append(f'<div style="color:#ff5f6d">M6 err: {e}</div>')

    return ('<div style="display:grid;grid-template-columns:repeat('
            'auto-fit,minmax(280px,1fr));gap:12px">'
            + "".join(cards)
            + "</div>")


# ── Individual milestone calculators ─────────────────────────────────


def _count_live_trades() -> int:
    """M1 — closed v7_okx_positions rows."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT COUNT(*) AS n FROM v7_okx_positions
                    WHERE status IN ('CLOSED', 'DEMOTED')
                """)
                return int((cur.fetchone() or {"n": 0})["n"] or 0)
            except Exception:
                return 0
    finally:
        conn.close()


def _days_since_last_unresolved_kill() -> float:
    """M2 — days since last kill_log entry with no resolution."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT MAX(ts) AS last_ts
                    FROM v7_okx_kill_log
                    WHERE resolved_at IS NULL
                """)
                row = cur.fetchone() or {}
                last = row.get("last_ts")
            except Exception:
                return 0
    finally:
        conn.close()
    if last is None:
        # No kill ever fired — count days since executor started or 0 if never
        return _days_since_executor_start()
    delta = (datetime.utcnow() - last).total_seconds() / 86400
    return max(0, delta)


def _days_since_executor_start() -> float:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT MIN(ts) AS first_ts
                    FROM v7_okx_executor_status
                """)
                row = cur.fetchone() or {}
                first = row.get("first_ts")
            except Exception:
                return 0
    finally:
        conn.close()
    # `ts` column doesn't exist on executor_status; the column is
    # last_changed_at. Recover gracefully:
    if first is None:
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MIN(last_changed_at) AS first_ts
                    FROM v7_okx_executor_status
                """)
                row = cur.fetchone() or {}
                first = row.get("first_ts")
            conn.close()
        except Exception:
            first = None
    if first is None:
        return 0
    return max(0, (datetime.utcnow() - first).total_seconds() / 86400)


def _consecutive_strong_ic_months(threshold: float = 0.10) -> int:
    """M3 — consecutive recent months with monthly IC >= threshold.

    Uses indicator_history (model pred + close); computes per-month
    Spearman IC between pred_return_4h and realised next-4h return
    derived from close prices.
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT dt, close, pred_return_4h
                    FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 180 DAY)
                    ORDER BY dt ASC
                """)
                rows = cur.fetchall()
            except Exception:
                return 0
    finally:
        conn.close()
    if not rows or len(rows) < 30:
        return 0
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    df["pred_return_4h"] = df["pred_return_4h"].astype(float)
    df["fwd_mean_4h"] = (df["close"].shift(-1)
                          .rolling(4).mean()
                          .shift(-(4 - 1)))
    df["y_4h"] = df["fwd_mean_4h"] / df["close"] - 1
    df = df.dropna(subset=["pred_return_4h", "y_4h"])
    if df.empty:
        return 0
    df["month"] = df["dt"].dt.to_period("M")
    consec = 0
    monthly = []
    for mo, g in df.groupby("month"):
        if len(g) < 50 or g["pred_return_4h"].std() < 1e-10:
            continue
        ic, _ = spearmanr(g["pred_return_4h"], g["y_4h"])
        monthly.append((mo, ic))
    # Count consecutive from most recent backwards
    for mo, ic in reversed(monthly):
        if not np.isnan(ic) and ic >= threshold:
            consec += 1
        else:
            break
    return consec


def _expected_bar_coverage_pct(days: int = 30) -> float:
    """M4 — fraction of expected hourly bars actually present."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT COUNT(*) AS n FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL %s DAY)
                """, (int(days),))
                actual = int((cur.fetchone() or {"n": 0})["n"] or 0)
            except Exception:
                return 0.0
    finally:
        conn.close()
    expected = days * 24
    return min(100.0, actual / expected * 100 if expected else 0)


def _distinct_regimes_in_live_window() -> int:
    """M5 — number of distinct regime codes observed during live."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT MIN(last_changed_at) AS exec_start
                    FROM v7_okx_executor_status
                """)
                row = cur.fetchone() or {}
                start = row.get("exec_start")
                if start is None:
                    return 0
                cur.execute("""
                    SELECT DISTINCT regime_code
                    FROM indicator_history
                    WHERE dt >= %s AND regime_code IS NOT NULL
                """, (start,))
                rows = cur.fetchall() or []
                # Only count non-WARMUP regimes (code -2 is WARMUP)
                non_warmup = [r for r in rows
                              if r.get("regime_code") not in (None, -2.0)]
                return len(non_warmup)
            except Exception:
                return 0
    finally:
        conn.close()


def _kill_recovery_count() -> int:
    """M6 — kill triggers that fired then auto-recovered (resolved)."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT COUNT(*) AS n
                    FROM v7_okx_kill_log
                    WHERE resolved_at IS NOT NULL
                """)
                return int((cur.fetchone() or {"n": 0})["n"] or 0)
            except Exception:
                return 0
    finally:
        conn.close()


# ── Kill recovery history ────────────────────────────────────────────


def _build_kill_recovery() -> str:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT ts, trigger_id, severity, context, resolved_at
                    FROM v7_okx_kill_log
                    ORDER BY ts DESC LIMIT 20
                """)
                rows = cur.fetchall() or []
            except Exception as e:
                return (f'<div style="color:rgba(154,160,166,0.6)">'
                        f'kill_log 載入失敗: {e}</div>')
    finally:
        conn.close()
    if not rows:
        return ('<div style="color:rgba(0,204,128,0.7);font-weight:500">'
                '✓ 系統啟動以來 0 個 kill trigger '
                '— 但這代表也沒實證過 recovery 路徑 (M6 待解鎖)</div>')

    items = []
    for r in rows:
        resolved = r.get("resolved_at")
        ic = "✓" if resolved else "⚠"
        col = "#36ffae" if resolved else "#f5b544"
        recovery_str = (f"resolved at {fmt_tpe(resolved)}"
                        if resolved else "<i>unresolved</i>")
        items.append(
            f'<tr><td style="color:{col}">{ic}</td>'
            f'<td>{fmt_tpe(r["ts"]) if r["ts"] else "?"}</td>'
            f'<td><b>{r.get("trigger_id","?")}</b></td>'
            f'<td>{r.get("severity","?")}</td>'
            f'<td style="font-size:10px">{recovery_str}</td></tr>'
        )
    return f"""
    <table>
      <tr><th></th><th>時間</th><th>Trigger</th><th>Severity</th>
          <th>Recovery</th></tr>
      {''.join(items)}
    </table>"""


# ── Uptime detail ────────────────────────────────────────────────────


def _build_uptime() -> str:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT DATE(dt) AS d, COUNT(*) AS n
                    FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 14 DAY)
                    GROUP BY DATE(dt) ORDER BY d DESC
                """)
                rows = cur.fetchall() or []
            except Exception:
                return ('<div style="color:rgba(154,160,166,0.6)">'
                        'uptime 數據載入失敗</div>')
    finally:
        conn.close()
    if not rows:
        return '<div style="color:rgba(154,160,166,0.6)">無 update_cycle 紀錄</div>'

    items = []
    for r in rows:
        d, n = r["d"], int(r["n"])
        if n >= 23:
            col, ic = "#36ffae", "✓"
        elif n >= 18:
            col, ic = "#f5b544", "⚠"
        else:
            col, ic = "#ff5f6d", "✗"
        items.append(
            f'<tr><td style="color:{col}">{ic}</td>'
            f'<td>{d}</td><td>{n}/24 hourly bars</td>'
            f'<td style="font-size:10px">'
            f'{"OK" if n >= 23 else ("partial" if n >= 18 else "GAP")}</td></tr>'
        )
    total_n = sum(int(r["n"]) for r in rows)
    expected = len(rows) * 24
    pct = total_n / expected * 100 if expected else 0
    color = "#36ffae" if pct >= 95 else ("#f5b544" if pct >= 80 else "#ff5f6d")
    return f"""
    <div style="color:rgba(154,160,166,0.8);font-size:12px;margin-bottom:6px">
      {len(rows)} 天平均 update_cycle 覆蓋率:
      <span style="color:{color};font-weight:600">{pct:.1f}%</span>
      ({total_n}/{expected} 預期 hourly bars)
    </div>
    <table>
      <tr><th></th><th>日期</th><th>覆蓋</th><th>狀態</th></tr>
      {''.join(items)}
    </table>"""
