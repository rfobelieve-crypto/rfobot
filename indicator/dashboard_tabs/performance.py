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
        section("📊 專業指標（vs BTC · payoff · 多空 · 成本）", "okxpro", True,
                _build_okx_pro_metrics()),
        section("📈 LIVE 淨值曲線（補資後 6/7 起 · vs BTC）", "okxequity", True,
                _build_okx_equity_chart()),
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


# ── LIVE Equity Curve (post-refund 6/7) ──────────────────────────────

def _build_okx_equity_chart() -> str:
    """Live wallet equity curve since the 2026-06-07 executor restart.

    Daily-close equity from v7_okx_balance_snapshots — the real post-refund
    account trajectory (NOT signal returns). Dashed baseline at the $105.15
    refund so above/below is obvious.
    """
    from indicator.okx.report import (
        EXECUTOR_RESTART_CAPITAL_USD as BASE,
        EXECUTOR_RESTART_SINCE as SINCE,
    )
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts, total_eq_usd FROM v7_okx_balance_snapshots "
                "WHERE id IN (SELECT MAX(id) FROM v7_okx_balance_snapshots "
                "             WHERE ts >= %s GROUP BY DATE(ts)) "
                "ORDER BY ts",
                (SINCE,))
            rows = cur.fetchall()
            cur.execute(
                "SELECT dt, close FROM indicator_history "
                "WHERE dt IN (SELECT MAX(dt) FROM indicator_history "
                "             WHERE dt >= %s GROUP BY DATE(dt)) "
                "ORDER BY dt",
                (SINCE,))
            btc_rows = cur.fetchall()
    except Exception as e:
        return f'<div style="color:rgba(154,160,166,0.5)">淨值曲線載入失敗: {e}</div>'
    finally:
        if conn:
            conn.close()

    if len(rows) < 2:
        return '<div style="color:rgba(154,160,166,0.5)">補資後淨值資料不足</div>'

    labels = [fmt_tpe(r["ts"], "%m/%d") if hasattr(r["ts"], "strftime")
              else str(r["ts"])[5:10] for r in rows]
    eq = [round(float(r["total_eq_usd"]), 2) for r in rows]
    cur_eq, peak = eq[-1], max(eq)
    delta = (cur_eq - BASE) / BASE * 100 if BASE else 0
    line_color = "#36ffae" if cur_eq >= BASE else "#ff5f6d"

    # BTC buy&hold normalised to $BASE at cohort start, aligned by date
    btc_map = {(r["dt"].date() if hasattr(r["dt"], "date") else str(r["dt"])[:10]):
               float(r["close"]) for r in btc_rows if r.get("close") is not None}
    btc_line, btc0, last = [], None, None
    for r in rows:
        dkey = r["ts"].date() if hasattr(r["ts"], "date") else str(r["ts"])[:10]
        v = btc_map.get(dkey, last)
        last = v if v is not None else last
        if btc0 is None and last is not None:
            btc0 = last
        btc_line.append(round(BASE * last / btc0, 2)
                        if (last is not None and btc0) else None)

    return f"""
    <div class="grid grid-3" style="margin-bottom:10px">
      {card("現值", f"${cur_eq:.2f}", f"{delta:+.2f}% vs ${BASE:.2f}", line_color)}
      {card("峰值", f"${peak:.2f}", "補資後高點", "#36ffae")}
      {card("起算", f"${BASE:.2f}", f"{SINCE} 補資")}
    </div>
    <div style="position:relative;height:180px">
      <canvas id="okxEquityChart"></canvas>
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('okxEquityChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [
            {{ label: '策略 Equity $', data: {_json.dumps(eq)},
               borderColor: '{line_color}', backgroundColor: 'rgba(255,255,255,0.04)',
               fill: true, tension: 0.25, borderWidth: 2, pointRadius: 2 }},
            {{ label: 'BTC 買入持有', data: {_json.dumps(btc_line)},
               borderColor: 'rgba(245,181,68,0.9)', backgroundColor: 'transparent',
               fill: false, tension: 0.25, borderWidth: 1.5, borderDash: [4,3], pointRadius: 0 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: true, labels: {{ color: 'rgba(154,160,166,0.85)', font: {{ size: 9 }}, boxWidth: 12 }} }},
            annotation: {{ annotations: {{
              base: {{ type: 'line', yMin: {BASE}, yMax: {BASE},
                       borderColor: 'rgba(154,160,166,0.45)', borderWidth: 1, borderDash: [5,5] }}
            }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: 'rgba(154,160,166,0.85)', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
            y: {{ ticks: {{ color: 'rgba(232,234,237,0.92)', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(255,255,255,0.06)' }},
                  title: {{ display: true, text: 'Equity $', color: 'rgba(154,160,166,0.8)' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Professional metrics (benchmark / payoff / long-short / cost / MDD) ─

def _build_okx_pro_metrics() -> str:
    """Pro live-trading metrics from the post-refund executor cohort:
    vs Buy&Hold BTC, payoff structure, long/short split, cost drag, MDD.
    """
    from indicator.okx.report import EXECUTOR_RESTART_SINCE as SINCE
    try:
        from indicator.okx.report import compute_okx_summary
        s = compute_okx_summary()
    except Exception as e:
        return f'<div style="color:#ff5f6d">專業指標載入失敗: {e}</div>'
    if not s.get("n_closed"):
        return ('<div style="color:rgba(154,160,166,0.6);font-size:12px">'
                '尚無已平倉 trade</div>')

    strat_ret = s.get("eq_pct_from_initial") or 0.0
    mdd = s.get("mdd_pct")
    bm = s.get("benchmark") or {}
    btc_ret, btc_mdd = bm.get("btc_ret_pct"), bm.get("btc_mdd_pct")
    payoff, pf = s.get("payoff_ratio"), s.get("profit_factor")
    aw, al = s.get("avg_win_pct", 0), s.get("avg_loss_pct", 0)
    cum_g, cum_n = s.get("cum_gross_pct", 0), s.get("cum_net_pct", 0)
    drag = s.get("cost_drag_bps", 0)
    ls, ss = s.get("long_stats", {}), s.get("short_stats", {})

    beat = (btc_ret is not None and strat_ret >= btc_ret)
    bm_color = "#36ffae" if beat else "#ff5f6d"
    btc_ret_str = f"{btc_ret:+.2f}%" if btc_ret is not None else "--"
    btc_mdd_str = f"{btc_mdd:.1f}%" if btc_mdd is not None else "--"
    payoff_str = f"{payoff:.2f}" if payoff is not None else "n/a"
    pf_str = f"{pf:.2f}" if pf is not None else "n/a"
    pf_color = "#36ffae" if (pf is not None and pf >= 1) else "#d9606a"

    def _side_card(label, st):
        c = "#36ffae" if st.get("cum_pct", 0) >= 0 else "#ff5f6d"
        return card(label, f"{st.get('cum_pct', 0):+.2f}%",
                    f"{st.get('n', 0)} 筆 · WR {st.get('wr', 0):.0f}%", c)

    return f"""
    <div style="color:rgba(154,160,166,0.75);font-size:11px;margin-bottom:6px">
      vs Buy&amp;Hold BTC（同期 {SINCE} 起）</div>
    <div class="grid grid-4" style="margin-bottom:12px">
      {card("策略報酬", f"{strat_ret:+.2f}%", f"BTC {btc_ret_str}", bm_color)}
      {card("策略 MDD", f"{mdd:.1f}%" if mdd is not None else "--", f"BTC {btc_mdd_str}", "#ff5f6d")}
      {card("Payoff", payoff_str, f"平均 {aw:+.2f}% / {al:+.2f}%")}
      {card("Profit Factor", pf_str, "總獲利 / 總虧損", pf_color)}
    </div>
    <div style="color:rgba(154,160,166,0.75);font-size:11px;margin-bottom:6px">
      多空拆分 · 成本拖累</div>
    <div class="grid grid-4">
      {_side_card("LONG", ls)}
      {_side_card("SHORT", ss)}
      {card("成本拖累", f"{drag:.1f} bps/筆", f"每筆 gross−net 差", "#f5b544")}
      {card("gross → net", f"{cum_n:+.2f}%", f"gross {cum_g:+.2f}%，費用吃 {cum_g - cum_n:+.2f}%")}
    </div>
    """
