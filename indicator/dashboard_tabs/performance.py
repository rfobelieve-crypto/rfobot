"""Tab 2: Model Performance — is the model still working?"""
from __future__ import annotations

import json as _json
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from indicator.dashboard_tabs._components import (
    card, section, status_dot, status_badge, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_performance() -> str:
    parts = [
        section("🔴 OKX LIVE Stage 3 ($100 + 10x)", "okxlive", True,
                _build_okx_live()),
        section("Alpha Decay Monitor", "decay", True, _build_alpha_decay()),
        section("IC / 勝率趨勢 (7 天)", "ictrend", True, _build_ic_trend()),
        section("信號累計曲線", "equity", True, _build_equity_curve()),
        section("V7 Paper (Shadow Baseline)", "v7paper", False,
                _build_v7_paper()),
        section("信心分佈 (48h)", "confdist", True, _build_confidence_dist()),
        section("預測 vs 實際 (24h)", "predva", True, _build_pred_vs_actual()),
        section("連續錯誤追蹤", "drawdown", True, _build_drawdown()),
        section("時段勝率熱力圖", "hourly_wr", True, _build_hourly_heatmap()),
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
        return f'<div style="color:#FF3366">OKX live cohort 載入失敗: {e}</div>'

    eq = s.get("current_equity_usd")
    avail = s.get("available_usd")
    initial = s.get("initial_capital_usd", 155.0)
    delta_pct = s.get("eq_pct_from_initial")
    age = s.get("balance_age_sec")
    status = s.get("executor_status") or "UNKNOWN"
    reason = s.get("executor_reason") or ""

    status_color = {
        "ACTIVE": "#00CC80",
        "HALTED": "#FFB400",
        "DEMOTED": "#FF3366",
        "INIT": "rgba(0,240,255,0.4)",
        "CONNECTING": "rgba(0,240,255,0.6)",
        "READY": "#00F0FF",
    }.get(status, "rgba(0,240,255,0.4)")

    # ── Row 1: account state
    if eq is not None:
        eq_color = "#00CC80" if (delta_pct or 0) >= 0 else "#FF3366"
        delta_str = f"{delta_pct:+.2f}% vs ${initial:.0f}"
        age_str = f"{age:.0f}s 前更新" if age is not None else "--"
        row1 = f"""
        <div class="grid grid-4" style="margin-bottom:12px">
          {card("Equity", f"${eq:.2f}", delta_str, eq_color)}
          {card("Available", f"${avail or 0:.2f}", age_str)}
          {card("Executor", status, reason[:40] if reason else "", status_color)}
          {card("Starting Capital", f"${initial:.0f}", "Stage 3 baseline")}
        </div>"""
    else:
        row1 = f"""
        <div class="grid grid-2" style="margin-bottom:12px">
          {card("Equity", "--", "尚無 balance snapshot")}
          {card("Executor", status, reason[:40] if reason else "", status_color)}
        </div>"""

    # ── Row 2: trade stats
    n_closed = s.get("n_closed", 0)
    if n_closed == 0:
        trade_block = (
            '<div style="color:rgba(0,240,255,0.5);font-size:12px;'
            'margin-bottom:10px">📊 <i>尚無已平倉 trade — '
            '等下一個 Strong/Moderate 訊號 + manual approval (/yes_1)</i></div>'
        )
    else:
        wr = s.get("win_rate_pct", 0)
        avg_bps = s.get("avg_net_bps", 0)
        cum_pct = s.get("cum_net_pct", 0)
        cum_eq_pct = s.get("cum_equity_pct", 0)
        pt_sharpe = s.get("sharpe_per_trade")
        ann_sharpe = s.get("sharpe_annualised")
        wr_color = "#00CC80" if wr >= 55 else "#CC4444"
        cum_color = "#00CC80" if cum_pct >= 0 else "#FF3366"
        sharpe_str = (f"per-trade {pt_sharpe:.2f}"
                       if pt_sharpe is not None else "n<2")
        sharpe_sub = (f"年化 {ann_sharpe:.2f}"
                       if ann_sharpe is not None else "")
        trade_block = f"""
        <div class="grid grid-4" style="margin-bottom:12px">
          {card("Closed Trades", str(n_closed), f"勝/敗 {s.get('wins',0)}/{n_closed - s.get('wins',0)}")}
          {card("勝率", f"{wr:.1f}%", "", wr_color)}
          {card("累計", f"{cum_pct:+.2f}%", f"equity Δ {cum_eq_pct:+.2f}%", cum_color)}
          {card("Sharpe", sharpe_str, sharpe_sub)}
        </div>"""

    # ── Row 3: open position
    op = s.get("open_position")
    if op:
        from datetime import datetime as _dt
        et = op.get("entry_time")
        held_h = ((_dt.utcnow() - et).total_seconds() / 3600
                  if et else 0)
        dir_color = "#00CC80" if op.get("direction") == "LONG" else "#FF3366"
        open_block = f"""
        <div style="color:rgba(0,240,255,0.55);font-size:11px;margin-bottom:6px">
          🔓 當前持倉 #{op.get('id')}
        </div>
        <div class="grid grid-4" style="margin-bottom:12px">
          {card(op.get('direction','?'), f"${op.get('entry_price',0):.1f}",
                op.get('entry_tier','--'), dir_color)}
          {card("Stop", f"${op.get('current_stop',0):.1f}", "")}
          {card("Size", f"{op.get('size_contracts',0)} contracts",
                f"notional ${op.get('notional_usd',0):.0f}")}
          {card("Held", f"{held_h:.1f}h", "time_cap 72h")}
        </div>"""
    else:
        open_block = (
            '<div style="color:rgba(0,240,255,0.4);font-size:12px;'
            'margin-bottom:10px">🔓 <i>flat — 無開倉</i></div>'
        )

    # ── Row 4: kill log alerts (7 days)
    kills = s.get("kill_log_7d") or []
    if kills:
        alert_lines = []
        for k in kills[:5]:
            ts = k.get("ts")
            ts_str = ts.strftime("%m/%d %H:%M") if hasattr(ts, "strftime") else str(ts)
            alert_lines.append(
                f'<tr><td style="color:#FF3366">⚠</td>'
                f'<td>{ts_str}</td>'
                f'<td><b>{k.get("trigger_id","?")}</b></td>'
                f'<td>{k.get("severity","?")}</td></tr>'
            )
        kill_block = f"""
        <div style="color:rgba(0,240,255,0.55);font-size:11px;margin-bottom:6px">
          ⚠️ Kill log (近 7 天 — {len(kills)} 筆)
        </div>
        <table>
          <tr><th></th><th>時間</th><th>Trigger</th><th>Severity</th></tr>
          {''.join(alert_lines)}
        </table>"""
    else:
        kill_block = (
            '<div style="color:rgba(0,204,128,0.6);font-size:12px;margin-top:8px">'
            '✓ 近 7 天無 kill trigger 觸發</div>'
        )

    return row1 + trade_block + open_block + kill_block


# ── V7 Paper Cohort ──────────────────────────────────────────────────

def _build_v7_paper() -> str:
    """V7 paper-trading cohort — backtest baseline + live forward stats."""
    try:
        from indicator.paper_trading import (
            compute_v7_summary, V7_BACKTEST_BASELINE,
        )
        s = compute_v7_summary()
    except Exception as e:
        return f'<div style="color:#FF3366">V7 cohort 載入失敗: {e}</div>'

    b = V7_BACKTEST_BASELINE
    bt = f"""
    <div style="color:rgba(0,240,255,0.55);font-size:11px;margin-bottom:6px">
      📦 Backtest 基準 — {b['config']} · WF-OOS · n={b['n']}
      <span style="color:rgba(255,180,0,0.7)">（{b['note']}）</span>
    </div>
    <div class="grid grid-4" style="margin-bottom:14px">
      {card("回測 ROI", f"{b['roi_pct']:+.1f}%", "5 個月")}
      {card("回測勝率", f"{b['wr_pct']:.1f}%", "")}
      {card("回測 MDD", f"{b['mdd_pct']:.1f}%", "")}
      {card("回測 Sharpe", f"{b['sharpe']:.2f}", f"每筆 {b['avg_eq_ret_pct']:+.2f}%")}
    </div>"""

    if s.get("empty"):
        live = ('<div style="color:rgba(0,240,255,0.4);font-size:12px">'
                '🤖 實盤 paper — 尚無已平倉 V7 訊號（與 LDC 並行中，'
                '等下一個 v7.1 Strong/Moderate 訊號開倉）</div>')
    else:
        o = s["overall"]
        roi_c = "#00CC80" if o["roi_pct"] >= 0 else "#FF3366"
        wr_c = "#00CC80" if o["wr"] >= 0.55 else "#CC4444"
        live = f"""
        <div style="color:rgba(0,240,255,0.55);font-size:11px;margin-bottom:6px">
          🤖 實盤 paper — 本金 $1000 · 2% 風險 / 1x / 複利 · n={o['n']}</div>
        <div class="grid grid-4">
          {card("實盤 ROI", f"{o['roi_pct']:+.1f}%", f"${o['final_equity']:,.0f}", roi_c)}
          {card("實盤勝率", f"{o['wr']*100:.1f}%", f"{o['n']} 筆", wr_c)}
          {card("實盤 MDD", f"{o['mdd_pct']:.1f}%", "")}
          {card("每筆均報", f"{o['avg_eq_ret_pct']:+.2f}%", f"持倉 {o['avg_hold_h']:.0f}h")}
        </div>"""
        r = s.get("recent", {})
        if r.get("n", 0) > 0:
            live += (f'<div style="color:rgba(0,240,255,0.5);font-size:11px;'
                     f'margin-top:8px">最近 {s["recent_window_days"]} 天: '
                     f'n={r["n"]} · WR {r["wr"]*100:.1f}% · '
                     f'每筆 {r["avg_eq_ret_pct"]:+.2f}%</div>')
    if s.get("n_shadow"):
        live += (f'<div style="color:rgba(255,180,0,0.65);font-size:11px;'
                 f'margin-top:6px">⚠ {s["n_shadow"]} 筆 shadow 已排除 '
                 f'— 模型轉換窗口 (前 48h)，不計入 cohort</div>')
    return bt + live


# ── Alpha Decay Monitor ──────────────────────────────────────────────

def _build_alpha_decay() -> str:
    try:
        from indicator.alpha_decay_monitor import run_full_check, STATUS_ICON
        results = run_full_check()
    except Exception as e:
        return f'<div style="color:#FF3366">Alpha Decay 載入失敗: {e}</div>'

    overall = results.get("overall", "unknown")
    ts = results.get("timestamp", "")

    signal_names = {
        "ic_trend": ("IC 趨勢", "滾動 IC 是否下降"),
        "importance_drift": ("特徵漂移", "Top-10 特徵是否穩定"),
        "churn_rate": ("信號翻轉", "方向預測是否頻繁反轉"),
        "confidence_wr": ("信心-勝率", "高信心是否=高勝率"),
        "signal_yield": ("Strong 產量", "Strong 信號比例趨勢"),
    }

    rows = []
    for key, (label, desc) in signal_names.items():
        r = results.get(key, {})
        status = r.get("status", "error")
        detail = r.get("detail", "N/A")
        rows.append(
            f"<tr><td>{status_dot(status)}</td>"
            f"<td><b>{label}</b><br><span style='color:rgba(0,240,255,0.5);font-size:10px'>{desc}</span></td>"
            f"<td>{status_badge(status)}</td>"
            f"<td style='font-size:11px'>{detail}</td></tr>"
        )

    return f"""
    <div style="margin-bottom:8px">
      整體狀態: {status_badge(overall)}
      <span style="color:rgba(0,240,255,0.5);font-size:11px;margin-left:8px">{ts}</span>
    </div>
    <table>
      <tr><th></th><th>信號</th><th>狀態</th><th>詳情</th></tr>
      {''.join(rows)}
    </table>"""


# ── IC / Win Rate Trend ──────────────────────────────────────────────

def _build_ic_trend() -> str:
    from scipy.stats import spearmanr

    try:
        from indicator.monitor_icir import DUAL_MODEL_START
    except ImportError:
        DUAL_MODEL_START = "2026-04-03"

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, pred_return_4h, pred_direction_code
                FROM indicator_history
                WHERE dt >= %s AND dt >= DATE_SUB(NOW(), INTERVAL 8 DAY)
                ORDER BY dt ASC
            """, (DUAL_MODEL_START,))
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">數據載入失敗: {e}</div>'

    if len(rows) < 30:
        return '<div style="color:rgba(0,240,255,0.3)">數據不足 (需要 30+ bars)</div>'

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1
    df = df.dropna(subset=["actual_4h", "pred_return_4h"])

    if len(df) < 30:
        return '<div style="color:rgba(0,240,255,0.3)">數據不足</div>'

    labels, ics, wrs = [], [], []
    window = 24
    for i in range(window, len(df), 6):
        chunk = df.iloc[i - window:i]
        if chunk["pred_return_4h"].std() < 1e-10:
            continue
        ic, _ = spearmanr(chunk["pred_return_4h"], chunk["actual_4h"])
        active = chunk[chunk["pred_direction_code"] != 0]
        if len(active) > 0:
            correct = ((active["pred_direction_code"] == 1) & (active["actual_4h"] > 0)) | \
                      ((active["pred_direction_code"] == -1) & (active["actual_4h"] < 0))
            wr = correct.mean() * 100
        else:
            wr = None

        labels.append(chunk["dt"].iloc[-1].strftime("%m/%d %H:%M"))
        ics.append(round(float(ic), 3) if not np.isnan(ic) else 0)
        wrs.append(round(float(wr), 1) if wr is not None else None)

    return f"""
    <div style="position:relative;height:180px">
      <canvas id="icTrendChart"></canvas>
    </div>
    <script>
    (function() {{
      var ctx = document.getElementById('icTrendChart').getContext('2d');
      new Chart(ctx, {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [
            {{ label: '滾動 IC (24h)', data: {_json.dumps(ics)},
               borderColor: '#00F0FF', backgroundColor: 'rgba(0,240,255,0.08)',
               yAxisID: 'y', tension: 0.3, borderWidth: 2, pointRadius: 1 }},
            {{ label: '勝率 % (24h)', data: {_json.dumps(wrs)},
               borderColor: '#00CC80', backgroundColor: 'rgba(0,204,128,0.08)',
               yAxisID: 'y1', tension: 0.3, borderWidth: 2, pointRadius: 1 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: 'rgba(0,240,255,0.85)', font: {{ size: 10 }} }} }},
            annotation: {{ annotations: {{
              zeroLine: {{ type: 'line', yMin: 0, yMax: 0, yScaleID: 'y',
                          borderColor: '#CC4444', borderWidth: 1, borderDash: [4,4] }}
            }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: 'rgba(0,240,255,0.6)', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: 'rgba(0,240,255,0.08)' }} }},
            y: {{ position: 'left', ticks: {{ color: '#00F0FF', font: {{ size: 9 }} }},
                  grid: {{ color: 'rgba(0,240,255,0.08)' }},
                  title: {{ display: true, text: 'IC', color: '#00F0FF' }} }},
            y1: {{ position: 'right', ticks: {{ color: '#00CC80', font: {{ size: 9 }} }},
                   grid: {{ drawOnChartArea: false }},
                   title: {{ display: true, text: '勝率 %', color: '#00CC80' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Strong Signal Equity Curve ───────────────────────────────────────

def _build_equity_curve() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, direction, actual_return_4h, correct, confidence
                FROM tracked_signals
                WHERE filled = 1 AND strength IN ('Strong', 'Moderate')
                ORDER BY signal_time ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">數據載入失敗: {e}</div>'

    if len(rows) < 3:
        return '<div style="color:rgba(0,240,255,0.3)">信號不足 3 筆</div>'

    labels, cum_ret = [], []
    total = 0
    wins, losses = 0, 0
    for r in rows:
        ret = float(r["actual_return_4h"] or 0)
        # Directional return: if DOWN signal, flip sign
        if r["direction"] == "DOWN":
            ret = -ret
        total += ret * 100  # in percentage points
        t = r["signal_time"]
        if hasattr(t, "strftime"):
            labels.append(t.strftime("%m/%d"))
        else:
            labels.append(str(t)[:10])
        cum_ret.append(round(total, 2))
        if r["correct"]:
            wins += 1
        else:
            losses += 1

    n = len(rows)
    wr = wins / n * 100 if n > 0 else 0
    final_color = "#00CC80" if total >= 0 else "#FF3366"

    return f"""
    <div class="grid grid-3" style="margin-bottom:10px">
      {card("總信號", str(n), f"勝: {wins} / 敗: {losses}")}
      {card("勝率", f"{wr:.1f}%", "", "#00CC80" if wr >= 60 else "#CC4444")}
      {card("累計回報", f"{total:+.1f}%", "方向性 paper return", final_color)}
    </div>
    <div style="position:relative;height:160px">
      <canvas id="equityChart"></canvas>
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('equityChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [{{ label: '累計回報 %', data: {_json.dumps(cum_ret)},
            borderColor: '{final_color}', backgroundColor: 'rgba(0,240,255,0.05)',
            fill: true, tension: 0.3, borderWidth: 2, pointRadius: 2 }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            annotation: {{ annotations: {{
              zero: {{ type: 'line', yMin: 0, yMax: 0,
                       borderColor: 'rgba(0,240,255,0.2)', borderWidth: 1, borderDash: [4,4] }}
            }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: 'rgba(0,240,255,0.6)', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,240,255,0.08)' }} }},
            y: {{ ticks: {{ color: 'rgba(0,240,255,0.85)', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,240,255,0.08)' }},
                  title: {{ display: true, text: '累計 %', color: 'rgba(0,240,255,0.5)' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Confidence Distribution ──────────────────────────────────────────

def _build_confidence_dist() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT confidence_score FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 48 HOUR)
                  AND confidence_score IS NOT NULL
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">{e}</div>'

    if not rows:
        return '<div style="color:rgba(0,240,255,0.3)">無數據</div>'

    scores = [float(r["confidence_score"]) for r in rows]

    # Build histogram buckets: 0-20, 20-40, 40-60, 60-80, 80-100
    buckets = [0] * 5
    bucket_labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    for s in scores:
        idx = min(int(s / 20), 4)
        buckets[idx] += 1

    colors = ["rgba(0,240,255,0.3)", "rgba(0,240,255,0.5)", "#CC4444", "#00F0FF", "#00CC80"]
    avg = sum(scores) / len(scores)
    median = sorted(scores)[len(scores) // 2]

    return f"""
    <div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:6px">
      平均: {avg:.1f} | 中位數: {median:.1f} | 樣本: {len(scores)}
    </div>
    <div style="position:relative;height:140px">
      <canvas id="confDistChart"></canvas>
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('confDistChart').getContext('2d'), {{
        type: 'bar',
        data: {{
          labels: {_json.dumps(bucket_labels)},
          datasets: [{{ data: {_json.dumps(buckets)},
            backgroundColor: {_json.dumps(colors)},
            borderRadius: 4 }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: 'rgba(0,240,255,0.6)', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
            y: {{ ticks: {{ color: 'rgba(0,240,255,0.6)', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,240,255,0.08)' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Prediction vs Actual ─────────────────────────────────────────────

def _build_pred_vs_actual() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, pred_direction_code, strength_code, confidence_score
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">{e}</div>'

    if not rows:
        return '<div style="color:rgba(0,240,255,0.3)">無數據</div>'

    labels, prices, colors_list, sizes = [], [], [], []
    for r in rows:
        dt = r["dt"]
        if hasattr(dt, "replace"):
            dt_local = dt.replace(tzinfo=timezone.utc).astimezone(TZ8)
        else:
            dt_local = dt
        labels.append(dt_local.strftime("%H:%M"))
        prices.append(round(float(r["close"]), 0))
        d = int(r["pred_direction_code"] or 0)
        s = int(r["strength_code"] or 1)
        colors_list.append("#00CC80" if d == 1 else "#FF3366" if d == -1 else "rgba(0,240,255,0.3)")
        sizes.append(6 if s == 3 else 4 if s == 2 else 2)

    return f"""
    <div style="position:relative;height:160px">
      <canvas id="predChart"></canvas>
    </div>
    <div style="color:rgba(0,240,255,0.5);font-size:10px;margin-top:4px">
      點: UP=綠, DOWN=紫, NEUTRAL=青 | 大點=Strong, 中點=Moderate
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('predChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [{{ label: 'BTC', data: {_json.dumps(prices)},
            borderColor: '#00F0FF', backgroundColor: 'rgba(0,240,255,0.05)',
            pointBackgroundColor: {_json.dumps(colors_list)},
            pointRadius: {_json.dumps(sizes)}, tension: 0.3, borderWidth: 2 }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: 'rgba(0,240,255,0.6)', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,240,255,0.08)' }} }},
            y: {{ ticks: {{ color: 'rgba(0,240,255,0.85)', font: {{ size: 9 }} }}, grid: {{ color: 'rgba(0,240,255,0.08)' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Drawdown (Consecutive Errors) ────────────────────────────────────

def _build_drawdown() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, direction, correct, strength, actual_return_4h
                FROM tracked_signals
                WHERE filled = 1
                ORDER BY signal_time ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">{e}</div>'

    if len(rows) < 5:
        return '<div style="color:rgba(0,240,255,0.3)">追蹤信號不足</div>'

    # Compute streaks
    current_streak = 0
    max_loss_streak = 0
    max_win_streak = 0
    current_type = None  # 'win' or 'loss'
    streaks = []

    for r in rows:
        is_correct = bool(r["correct"])
        if is_correct:
            if current_type == "win":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "win"
            max_win_streak = max(max_win_streak, current_streak)
        else:
            if current_type == "loss":
                current_streak += 1
            else:
                current_streak = 1
                current_type = "loss"
            max_loss_streak = max(max_loss_streak, current_streak)

    # Current active streak
    recent_streak = 0
    recent_type = None
    for r in reversed(rows):
        is_correct = bool(r["correct"])
        if recent_type is None:
            recent_type = "win" if is_correct else "loss"
            recent_streak = 1
        elif (is_correct and recent_type == "win") or (not is_correct and recent_type == "loss"):
            recent_streak += 1
        else:
            break

    streak_color = "#00CC80" if recent_type == "win" else "#FF3366"
    alert = ""
    if recent_type == "loss" and recent_streak >= max_loss_streak and recent_streak >= 3:
        alert = '<div style="color:#FF3366;font-weight:600;margin-top:6px">&#9888; 目前連敗次數已達歷史最高！</div>'

    return f"""
    <div class="grid grid-4">
      {card("當前連續", f'{recent_streak} {("連勝" if recent_type == "win" else "連敗")}',
            "", streak_color)}
      {card("歷史最長連勝", str(max_win_streak), "", "#00CC80")}
      {card("歷史最長連敗", str(max_loss_streak), "", "#FF3366")}
      {card("總信號數", str(len(rows)), "")}
    </div>
    {alert}
    """


# ── Hourly Win Rate Heatmap ──────────────────────────────────────────

def _build_hourly_heatmap() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT HOUR(signal_time) as hr, correct, COUNT(*) as cnt
                FROM tracked_signals
                WHERE filled = 1
                GROUP BY HOUR(signal_time), correct
                ORDER BY hr
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:rgba(0,240,255,0.3)">{e}</div>'

    if not rows:
        return '<div style="color:rgba(0,240,255,0.3)">數據不足</div>'

    # Build hour -> {wins, total}
    hours_data = {}
    for r in rows:
        hr = int(r["hr"])
        cnt = int(r["cnt"])
        is_win = bool(r["correct"])
        if hr not in hours_data:
            hours_data[hr] = {"wins": 0, "total": 0}
        hours_data[hr]["total"] += cnt
        if is_win:
            hours_data[hr]["wins"] += cnt

    # Build 24-hour grid (UTC+8)
    cells = []
    for h in range(24):
        utc_h = (h - 8) % 24  # convert display hour (UTC+8) back to UTC for lookup
        d = hours_data.get(utc_h, {"wins": 0, "total": 0})
        if d["total"] > 0:
            wr = d["wins"] / d["total"] * 100
            # Color: green if good, red if bad
            if wr >= 65:
                bg = "#00CC80"
            elif wr >= 50:
                bg = "#CC4444"
            else:
                bg = "#FF3366"
            opacity = min(0.3 + d["total"] / 20, 1.0)
            cells.append(
                f'<div class="hm-cell" style="background:{bg};opacity:{opacity:.2f}"'
                f' title="{h}:00 UTC+8 | WR={wr:.0f}% ({d["total"]} signals)">'
                f'{wr:.0f}</div>'
            )
        else:
            cells.append(f'<div class="hm-cell" style="background:#1A1A2E" title="{h}:00 UTC+8 | 無數據">-</div>')

    return f"""
    <div style="color:rgba(0,240,255,0.5);font-size:11px;margin-bottom:6px">每小時勝率 (UTC+8) — 顏色越綠越準</div>
    <div class="heatmap-grid">{''.join(cells)}</div>
    <div class="heatmap-labels">
      {''.join(f'<div>{h}</div>' for h in range(24))}
    </div>"""
