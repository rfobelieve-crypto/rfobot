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
        section("Alpha Decay Monitor", "decay", True, _build_alpha_decay()),
        section("IC / 勝率趨勢 (7 天)", "ictrend", True, _build_ic_trend()),
        section("信號累計曲線", "equity", True, _build_equity_curve()),
        section("信心分佈 (48h)", "confdist", True, _build_confidence_dist()),
        section("預測 vs 實際 (24h)", "predva", True, _build_pred_vs_actual()),
        section("連續錯誤追蹤", "drawdown", True, _build_drawdown()),
        section("時段勝率熱力圖", "hourly_wr", True, _build_hourly_heatmap()),
    ]
    return "\n".join(parts)


# ── Alpha Decay Monitor ──────────────────────────────────────────────

def _build_alpha_decay() -> str:
    try:
        from indicator.alpha_decay_monitor import run_full_check, STATUS_ICON
        results = run_full_check()
    except Exception as e:
        return f'<div style="color:#f44336">Alpha Decay 載入失敗: {e}</div>'

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
            f"<td><b>{label}</b><br><span style='color:#8b949e;font-size:10px'>{desc}</span></td>"
            f"<td>{status_badge(status)}</td>"
            f"<td style='font-size:11px'>{detail}</td></tr>"
        )

    return f"""
    <div style="margin-bottom:8px">
      整體狀態: {status_badge(overall)}
      <span style="color:#8b949e;font-size:11px;margin-left:8px">{ts}</span>
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
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if len(rows) < 30:
        return '<div style="color:#666">數據不足 (需要 30+ bars)</div>'

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1
    df = df.dropna(subset=["actual_4h", "pred_return_4h"])

    if len(df) < 30:
        return '<div style="color:#666">數據不足</div>'

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
               borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.08)',
               yAxisID: 'y', tension: 0.3, borderWidth: 2, pointRadius: 1 }},
            {{ label: '勝率 % (24h)', data: {_json.dumps(wrs)},
               borderColor: '#4caf50', backgroundColor: 'rgba(76,175,80,0.08)',
               yAxisID: 'y1', tension: 0.3, borderWidth: 2, pointRadius: 1 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }},
            annotation: {{ annotations: {{
              zeroLine: {{ type: 'line', yMin: 0, yMax: 0, yScaleID: 'y',
                          borderColor: '#ff9800', borderWidth: 1, borderDash: [4,4] }}
            }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ position: 'left', ticks: {{ color: '#58a6ff', font: {{ size: 9 }} }},
                  grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'IC', color: '#58a6ff' }} }},
            y1: {{ position: 'right', ticks: {{ color: '#4caf50', font: {{ size: 9 }} }},
                   grid: {{ drawOnChartArea: false }},
                   title: {{ display: true, text: '勝率 %', color: '#4caf50' }} }}
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
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if len(rows) < 3:
        return '<div style="color:#666">信號不足 3 筆</div>'

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
    final_color = "#4caf50" if total >= 0 else "#f44336"

    return f"""
    <div class="grid grid-3" style="margin-bottom:10px">
      {card("總信號", str(n), f"勝: {wins} / 敗: {losses}")}
      {card("勝率", f"{wr:.1f}%", "", "#4caf50" if wr >= 60 else "#ff9800")}
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
            borderColor: '{final_color}', backgroundColor: 'rgba(88,166,255,0.05)',
            fill: true, tension: 0.3, borderWidth: 2, pointRadius: 2 }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            annotation: {{ annotations: {{
              zero: {{ type: 'line', yMin: 0, yMax: 0,
                       borderColor: '#666', borderWidth: 1, borderDash: [4,4] }}
            }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: '累計 %', color: '#8b949e' }} }}
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
        return f'<div style="color:#666">{e}</div>'

    if not rows:
        return '<div style="color:#666">無數據</div>'

    scores = [float(r["confidence_score"]) for r in rows]

    # Build histogram buckets: 0-20, 20-40, 40-60, 60-80, 80-100
    buckets = [0] * 5
    bucket_labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    for s in scores:
        idx = min(int(s / 20), 4)
        buckets[idx] += 1

    colors = ["#666", "#8b949e", "#ff9800", "#58a6ff", "#4caf50"]
    avg = sum(scores) / len(scores)
    median = sorted(scores)[len(scores) // 2]

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
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
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
            y: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }}
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
        return f'<div style="color:#666">{e}</div>'

    if not rows:
        return '<div style="color:#666">無數據</div>'

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
        colors_list.append("#4caf50" if d == 1 else "#f44336" if d == -1 else "#666")
        sizes.append(6 if s == 3 else 4 if s == 2 else 2)

    return f"""
    <div style="position:relative;height:160px">
      <canvas id="predChart"></canvas>
    </div>
    <div style="color:#8b949e;font-size:10px;margin-top:4px">
      點: UP=綠, DOWN=紅, NEUTRAL=灰 | 大點=Strong, 中點=Moderate
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('predChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [{{ label: 'BTC', data: {_json.dumps(prices)},
            borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.05)',
            pointBackgroundColor: {_json.dumps(colors_list)},
            pointRadius: {_json.dumps(sizes)}, tension: 0.3, borderWidth: 2 }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }}
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
        return f'<div style="color:#666">{e}</div>'

    if len(rows) < 5:
        return '<div style="color:#666">追蹤信號不足</div>'

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

    streak_color = "#4caf50" if recent_type == "win" else "#f44336"
    alert = ""
    if recent_type == "loss" and recent_streak >= max_loss_streak and recent_streak >= 3:
        alert = '<div style="color:#f44336;font-weight:600;margin-top:6px">&#9888; 目前連敗次數已達歷史最高！</div>'

    return f"""
    <div class="grid grid-4">
      {card("當前連續", f'{recent_streak} {("連勝" if recent_type == "win" else "連敗")}',
            "", streak_color)}
      {card("歷史最長連勝", str(max_win_streak), "", "#4caf50")}
      {card("歷史最長連敗", str(max_loss_streak), "", "#f44336")}
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
        return f'<div style="color:#666">{e}</div>'

    if not rows:
        return '<div style="color:#666">數據不足</div>'

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
                bg = "#4caf50"
            elif wr >= 50:
                bg = "#ff9800"
            else:
                bg = "#f44336"
            opacity = min(0.3 + d["total"] / 20, 1.0)
            cells.append(
                f'<div class="hm-cell" style="background:{bg};opacity:{opacity:.2f}"'
                f' title="{h}:00 UTC+8 | WR={wr:.0f}% ({d["total"]} signals)">'
                f'{wr:.0f}</div>'
            )
        else:
            cells.append(f'<div class="hm-cell" style="background:#21262d" title="{h}:00 UTC+8 | 無數據">-</div>')

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">每小時勝率 (UTC+8) — 顏色越綠越準</div>
    <div class="heatmap-grid">{''.join(cells)}</div>
    <div class="heatmap-labels">
      {''.join(f'<div>{h}</div>' for h in range(24))}
    </div>"""
