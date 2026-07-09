"""Tab 2: Model Performance — is the model still working?"""
from __future__ import annotations

import json as _json
import logging
from datetime import datetime, timezone, timedelta

from indicator.timeutil import fmt_tpe

import numpy as np
import pandas as pd

from indicator.dashboard_tabs._components import (
    card, section, status_dot, status_badge, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_performance() -> str:
    parts = [
        section("🔴 OKX LIVE Stage 3 · 2x 有效槓桿（補資後起算）", "okxlive", True,
                _build_okx_live()),
    ]
    return "\n".join(parts)


# ── OKX LIVE Stage 3 cohort ──────────────────────────────────────────


def _build_okx_live() -> str:
    """Live OKX trading state — equity, executor status, Sharpe, trades.

    Pulls from indicator.okx.report.compute_okx_summary which reads the
    v7_okx_* tables populated by the production executor on Railway.
    """
    try:
        from indicator.okx.report import compute_okx_summary
        s = compute_okx_summary()
    except Exception as e:
        return f'<div style="color:#ff5f6d">OKX live cohort 載入失敗: {e}</div>'

    eq = s.get("current_equity_usd")
    avail = s.get("available_usd")
    initial = s.get("initial_capital_usd", 155.0)
    basis_since = s.get("capital_basis_since")
    delta_pct = s.get("eq_pct_from_initial")
    age = s.get("balance_age_sec")
    status = s.get("executor_status") or "UNKNOWN"
    reason = s.get("executor_reason") or ""

    status_color = {
        "ACTIVE": "#36ffae",
        "HALTED": "#f5b544",
        "DEMOTED": "#ff5f6d",
        "INIT": "rgba(154,160,166,0.6)",
        "CONNECTING": "rgba(154,160,166,0.85)",
        "READY": "#36ffae",
    }.get(status, "rgba(154,160,166,0.6)")

    # ── Row 1: account state
    if eq is not None:
        eq_color = "#36ffae" if (delta_pct or 0) >= 0 else "#ff5f6d"
        delta_str = f"{delta_pct:+.2f}% vs ${initial:.2f}"
        age_str = f"{age:.0f}s 前更新" if age is not None else "--"
        row1 = f"""
        <div class="grid grid-4" style="margin-bottom:12px">
          {card("Equity", f"${eq:.2f}", delta_str, eq_color)}
          {card("Available", f"${avail or 0:.2f}", age_str)}
          {card("Executor", status, reason[:40] if reason else "", status_color)}
          {card("Starting Capital", f"${initial:.2f}",
                f"executor 補資後 ({basis_since})" if basis_since else "Stage 3 baseline")}
        </div>"""
    else:
        row1 = f"""
        <div class="grid grid-2" style="margin-bottom:12px">
          {card("Equity", "--", "尚無 balance snapshot")}
          {card("Executor", status, reason[:40] if reason else "", status_color)}
        </div>"""

    # ── Row 2: trade stats — three WR definitions surfaced side by side
    # so the gap between "price direction" and "wallet truth" is visible
    n_closed = s.get("n_closed", 0)
    if n_closed == 0:
        trade_block = (
            '<div style="color:rgba(154,160,166,0.8);font-size:12px;'
            'margin-bottom:10px">📊 <i>尚無已平倉 trade — '
            '等下一個 Strong/Moderate 訊號 + manual approval (/yes_1)</i></div>'
        )
    else:
        wr_gross = s.get("win_rate_pct", 0)
        wr_net = s.get("win_rate_pct_net", wr_gross)
        wr_eq = s.get("win_rate_pct_equity", wr_gross)
        avg_bps = s.get("avg_net_bps", 0)
        cum_pct = s.get("cum_net_pct", 0)
        cum_eq_pct = s.get("cum_equity_pct", 0)
        pt_sharpe = s.get("sharpe_per_trade")
        ann_sharpe = s.get("sharpe_annualised")
        # Color WR by equity (wallet truth) — that's the one that matters
        wr_color = "#36ffae" if wr_eq >= 55 else "#d9606a"
        cum_color = "#36ffae" if cum_pct >= 0 else "#ff5f6d"
        sharpe_str = (f"per-trade {pt_sharpe:.2f}"
                       if pt_sharpe is not None else "n<2")
        sharpe_sub = (f"年化 {ann_sharpe:.2f}"
                       if ann_sharpe is not None else "")
        wr_three = (
            f"<div style='font-size:10px;line-height:1.45;color:rgba(154,160,166,0.85)'>"
            f"gross {wr_gross:.0f}% &nbsp;·&nbsp; "
            f"net {wr_net:.0f}% &nbsp;·&nbsp; "
            f"<b style='color:{wr_color}'>equity {wr_eq:.0f}%</b>"
            f"</div>"
        )
        trade_block = f"""
        <div class="grid grid-4" style="margin-bottom:12px">
          {card("Closed Trades", str(n_closed),
                f"勝/敗(equity) {s.get('wins_equity', s.get('wins', 0))}/"
                f"{n_closed - s.get('wins_equity', s.get('wins', 0))}")}
          {card("勝率三層", f"{wr_eq:.0f}%", wr_three, wr_color)}
          {card("累計", f"{cum_pct:+.2f}%", f"equity Δ {cum_eq_pct:+.2f}%", cum_color)}
          {card("Sharpe", sharpe_str, sharpe_sub)}
        </div>"""

    # ── Row 2.5: Gate B progress (Stage 3 → 4a promotion gate)
    gate_b_block = _build_gate_b_card(s.get("gate_b"))

    # ── Row 3: open position
    op = s.get("open_position")
    if op:
        from datetime import datetime as _dt
        et = op.get("entry_time")
        held_h = ((_dt.utcnow() - et).total_seconds() / 3600
                  if et else 0)
        dir_color = "#36ffae" if op.get("direction") == "LONG" else "#ff5f6d"
        # time_cap removed 2026-06-10 — exits now come from 3xATR trail or
        # opposite signal only. No fixed time horizon to display.
        open_block = f"""
        <div style="color:rgba(154,160,166,0.8);font-size:11px;margin-bottom:6px">
          🔓 當前持倉 #{op.get('id')}
        </div>
        <div class="grid grid-4" style="margin-bottom:12px">
          {card(op.get('direction','?'), f"${op.get('entry_price',0):.1f}",
                op.get('entry_tier','--'), dir_color)}
          {card("Stop", f"${op.get('current_stop',0):.1f}", "3×ATR trailing")}
          {card("Size", f"{op.get('size_contracts',0)} contracts",
                f"notional ${op.get('notional_usd',0):.0f}")}
          {card("Held", f"{held_h:.1f}h", "no time cap")}
        </div>"""
    else:
        open_block = (
            '<div style="color:rgba(154,160,166,0.6);font-size:12px;'
            'margin-bottom:10px">🔓 <i>flat — 無開倉</i></div>'
        )

    # ── Row 4: kill log alerts (7 days)
    kills = s.get("kill_log_7d") or []
    if kills:
        alert_lines = []
        for k in kills[:5]:
            ts = k.get("ts")
            ts_str = fmt_tpe(ts) if hasattr(ts, "strftime") else str(ts)
            alert_lines.append(
                f'<tr><td style="color:#ff5f6d">⚠</td>'
                f'<td>{ts_str}</td>'
                f'<td><b>{k.get("trigger_id","?")}</b></td>'
                f'<td>{k.get("severity","?")}</td></tr>'
            )
        kill_block = f"""
        <div style="color:rgba(154,160,166,0.8);font-size:11px;margin-bottom:6px">
          ⚠️ Kill log (近 7 天 — {len(kills)} 筆)
        </div>
        <table>
          <tr><th></th><th>時間</th><th>Trigger</th><th>Severity</th></tr>
          {''.join(alert_lines)}
        </table>"""
    else:
        kill_block = (
            '<div style="color:rgba(54,255,174,0.6);font-size:12px;margin-top:8px">'
            '✓ 近 7 天無 kill trigger 觸發</div>'
        )

    return row1 + trade_block + gate_b_block + open_block + kill_block


def _build_gate_b_card(gb: dict | None) -> str:
    """Stage 3 → 4a promotion gate progress.

    Compact card showing:
      - Progress bar (n_closed / sample_target)
      - Pass criteria status (accumulating / marginal / passed / failed)
      - Bootstrap 95% CI on avg net bps (the actual judgement metric)
    """
    if not gb:
        return ""
    status = gb.get("status", "accumulating")
    n = gb.get("n_closed", 0)
    target = gb.get("sample_target", 50)
    sample_min = gb.get("sample_min", 30)
    progress = gb.get("progress_pct", 0.0)
    avg = gb.get("avg_net_bps")
    ci_lo = gb.get("ci_lo_bps")
    ci_hi = gb.get("ci_hi_bps")

    icon, color = {
        "accumulating": ("🟡", "#f5b544"),
        "marginal":     ("🟠", "#f5b544"),
        "passed":       ("🟢", "#36ffae"),
        "failed":       ("🔴", "#ff5f6d"),
    }.get(status, ("⚪", "rgba(154,160,166,0.6)"))

    bar_fill = min(progress, 100.0)
    bar = (
        f"<div style='background:rgba(255,255,255,0.06);height:6px;"
        f"border-radius:3px;overflow:hidden;margin:4px 0'>"
        f"<div style='background:{color};height:100%;width:{bar_fill:.0f}%'></div>"
        f"</div>"
    )

    ci_str = (
        f"95% CI [{ci_lo:+.1f}, {ci_hi:+.1f}] bps"
        if (ci_lo is not None and ci_hi is not None)
        else "n&lt;5 — CI 未計算"
    )
    avg_str = f"{avg:+.1f} bps" if avg is not None else "--"

    pass_criteria = (
        f"通過條件：n ≥ {sample_min} + avg net &gt; 0 + 95% CI 下緣 &gt; 0"
    )

    return f"""
    <div style="color:rgba(154,160,166,0.85);font-size:11px;margin-bottom:6px">
      {icon} Gate B (Stage 3 → 4a 升階閘) — {gb.get('summary','')}
    </div>
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                border-radius:6px;padding:10px 14px;margin-bottom:12px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;
                  font-size:11px;color:rgba(154,160,166,0.85)">
        <span>進度</span>
        <span><b style="color:{color}">{n}</b> / {sample_min}–{target} 筆</span>
      </div>
      {bar}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;
                  margin-top:8px;font-size:11px">
        <div>
          <div style="color:rgba(154,160,166,0.75)">Avg net</div>
          <div style="color:{color};font-weight:600">{avg_str}</div>
        </div>
        <div>
          <div style="color:rgba(154,160,166,0.75)">Bootstrap</div>
          <div style="color:rgba(232,234,237,0.92)">{ci_str}</div>
        </div>
      </div>
      <div style="font-size:10px;color:rgba(154,160,166,0.65);margin-top:6px">
        {pass_criteria}
      </div>
    </div>
    """
