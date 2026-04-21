"""Tab: Analytics — deep-dive visualizations for signal quality and model behaviour."""
from __future__ import annotations

import json as _json
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from indicator.dashboard_tabs._components import (
    card, section, gauge_bar, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_analytics(state: dict, engine) -> str:
    parts = [
        section("Signal Heatmap (30d)", "sigHeatmap", True, _build_signal_heatmap()),
        section("Equity Curve by Tier", "eqTier", True, _build_equity_by_tier()),
        section("Rolling IC (7d / 30d)", "rollingIc", True, _build_rolling_ic()),
        section("Pred vs Actual Scatter (7d)", "predScatter", True, _build_scatter()),
        section("Feature Radar (Z-Score)", "featRadar", True,
                _build_feature_radar(state, engine)),
        section("Regime + Signal Overlay (7d)", "regimeSig", True,
                _build_regime_signals()),
        section("Real-time Gauges", "rtGauges", True, _build_gauges()),
        section("Signal on Price (72h)", "sigPrice", True,
                _build_signal_price_chart()),
    ]
    return "\n".join(parts)


# ── 1. Signal Heatmap ──────────────────────────────────────────────────

def _build_signal_heatmap() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, confidence, correct, filled
                FROM tracked_signals
                WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                ORDER BY signal_time ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if not rows:
        return '<div style="color:#666">數據不足</div>'

    # Build grid: date (x) x confidence bucket (y)
    buckets = ["65-70", "70-75", "75-80", "80-85", "85+"]

    def _bucket_idx(conf: float) -> int:
        if conf >= 85:
            return 4
        elif conf >= 80:
            return 3
        elif conf >= 75:
            return 2
        elif conf >= 70:
            return 1
        return 0

    # Collect unique dates and cells
    date_set: dict[str, int] = {}  # date_str -> col index
    cells: list[dict] = []

    for r in rows:
        conf = float(r["confidence"] or 0)
        if conf < 65:
            continue
        t = r["signal_time"]
        if hasattr(t, "replace"):
            t_local = t.replace(tzinfo=timezone.utc).astimezone(TZ8)
            date_str = t_local.strftime("%m/%d")
        else:
            date_str = str(t)[:10]

        if date_str not in date_set:
            date_set[date_str] = len(date_set)

        filled = bool(r["filled"])
        correct = bool(r["correct"]) if filled else None

        if correct is True:
            color = "rgba(76,175,80,0.8)"
        elif correct is False:
            color = "rgba(244,67,54,0.8)"
        else:
            color = "rgba(139,148,158,0.5)"

        cells.append({
            "x": date_set[date_str],
            "y": _bucket_idx(conf),
            "r": 8,
            "bg": color,
        })

    if not cells:
        return '<div style="color:#666">無符合條件的信號 (confidence >= 65)</div>'

    date_labels = list(date_set.keys())

    # Build as bubble chart
    # Group cells by color for separate datasets
    green_pts = [{"x": c["x"], "y": c["y"], "r": c["r"]}
                 for c in cells if "76,175,80" in c["bg"]]
    red_pts = [{"x": c["x"], "y": c["y"], "r": c["r"]}
               for c in cells if "244,67,54" in c["bg"]]
    grey_pts = [{"x": c["x"], "y": c["y"], "r": c["r"]}
                for c in cells if "139,148,158" in c["bg"]]

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      綠=正確, 紅=錯誤, 灰=待定 | Y 軸=信心分數區間
    </div>
    <div style="position:relative;height:200px">
      <canvas id="signalHeatmap"></canvas>
    </div>
    <script>
    (function(){{
      new Chart(document.getElementById('signalHeatmap').getContext('2d'), {{
        type: 'bubble',
        data: {{
          datasets: [
            {{ label: '正確', data: {_json.dumps(green_pts)},
               backgroundColor: 'rgba(76,175,80,0.8)' }},
            {{ label: '錯誤', data: {_json.dumps(red_pts)},
               backgroundColor: 'rgba(244,67,54,0.8)' }},
            {{ label: '待定', data: {_json.dumps(grey_pts)},
               backgroundColor: 'rgba(139,148,158,0.5)' }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }}
          }},
          scales: {{
            x: {{
              type: 'linear', min: -0.5, max: {len(date_labels) - 0.5},
              ticks: {{
                color: '#8b949e', font: {{ size: 9 }},
                callback: function(v) {{
                  var labels = {_json.dumps(date_labels)};
                  return labels[Math.round(v)] || '';
                }}
              }},
              grid: {{ color: '#21262d' }}
            }},
            y: {{
              type: 'linear', min: -0.5, max: 4.5,
              ticks: {{
                color: '#8b949e', font: {{ size: 9 }},
                callback: function(v) {{
                  var b = {_json.dumps(buckets)};
                  return b[Math.round(v)] || '';
                }}
              }},
              grid: {{ color: '#21262d' }}
            }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 2. Equity Curve by Tier ────────────────────────────────────────────

def _build_equity_by_tier() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, direction, strength, actual_return_4h
                FROM tracked_signals
                WHERE filled = 1
                ORDER BY signal_time ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if len(rows) < 3:
        return '<div style="color:#666">信號不足 3 筆</div>'

    labels_s, cum_strong = [], []
    labels_m, cum_moderate = [], []
    total_s, total_m = 0.0, 0.0

    for r in rows:
        ret = float(r["actual_return_4h"] or 0)
        if r["direction"] == "DOWN":
            ret = -ret
        ret_pct = ret * 100

        t = r["signal_time"]
        lbl = t.strftime("%m/%d %H:%M") if hasattr(t, "strftime") else str(t)[:16]

        if r["strength"] == "Strong":
            total_s += ret_pct
            labels_s.append(lbl)
            cum_strong.append(round(total_s, 2))
        elif r["strength"] == "Moderate":
            total_m += ret_pct
            labels_m.append(lbl)
            cum_moderate.append(round(total_m, 2))

    # Merge into a single timeline
    all_labels: list[str] = []
    s_vals: list = []
    m_vals: list = []
    s_idx, m_idx = 0, 0
    s_cur, m_cur = 0.0, 0.0

    # Combine both timelines in order
    combined = []
    for lbl, val in zip(labels_s, cum_strong):
        combined.append((lbl, "S", val))
    for lbl, val in zip(labels_m, cum_moderate):
        combined.append((lbl, "M", val))
    combined.sort(key=lambda x: x[0])

    last_s, last_m = 0.0, 0.0
    for lbl, tier, val in combined:
        all_labels.append(lbl)
        if tier == "S":
            last_s = val
        else:
            last_m = val
        s_vals.append(last_s)
        m_vals.append(last_m)

    if not all_labels:
        return '<div style="color:#666">無已填入信號</div>'

    s_color = "#f44336"
    m_color = "#ff9800"

    return f"""
    <div class="grid grid-3" style="margin-bottom:8px">
      {card("Strong 累計", f'{last_s:+.2f}%', f'{len(cum_strong)} 筆',
            "#4caf50" if last_s >= 0 else "#f44336")}
      {card("Moderate 累計", f'{last_m:+.2f}%', f'{len(cum_moderate)} 筆',
            "#4caf50" if last_m >= 0 else "#f44336")}
      {card("總信號", str(len(rows)), "")}
    </div>
    <div style="position:relative;height:180px">
      <canvas id="equityTier"></canvas>
    </div>
    <script>
    (function(){{
      new Chart(document.getElementById('equityTier').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(all_labels)},
          datasets: [
            {{ label: 'Strong', data: {_json.dumps(s_vals)},
               borderColor: '{s_color}', backgroundColor: 'rgba(244,67,54,0.05)',
               fill: false, tension: 0.3, borderWidth: 2, pointRadius: 2 }},
            {{ label: 'Moderate', data: {_json.dumps(m_vals)},
               borderColor: '{m_color}', backgroundColor: 'rgba(255,152,0,0.05)',
               fill: false, tension: 0.3, borderWidth: 2, pointRadius: 2 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: '累計回報 %', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 3. Rolling IC Chart ────────────────────────────────────────────────

def _build_rolling_ic() -> str:
    from scipy.stats import spearmanr

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, pred_return_4h
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 31 DAY)
                  AND pred_return_4h IS NOT NULL
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if len(rows) < 30:
        return '<div style="color:#666">數據不足 (需要 30+ bars)</div>'

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    df["pred_return_4h"] = df["pred_return_4h"].astype(float)
    df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1
    df = df.dropna(subset=["actual_4h", "pred_return_4h"])

    if len(df) < 30:
        return '<div style="color:#666">數據不足</div>'

    window_7d = 7 * 24   # hours
    window_30d = 30 * 24

    labels: list[str] = []
    ic_7d: list[float | None] = []
    ic_30d: list[float | None] = []

    step = max(1, len(df) // 80)  # ~80 points on chart

    for i in range(window_7d, len(df), step):
        dt_str = df["dt"].iloc[i].strftime("%m/%d")
        labels.append(dt_str)

        # 7d window
        chunk_7 = df.iloc[max(0, i - window_7d):i]
        if len(chunk_7) >= 20 and chunk_7["pred_return_4h"].std() > 1e-10:
            ic, _ = spearmanr(chunk_7["pred_return_4h"], chunk_7["actual_4h"])
            ic_7d.append(round(float(ic), 3) if not np.isnan(ic) else None)
        else:
            ic_7d.append(None)

        # 30d window
        chunk_30 = df.iloc[max(0, i - window_30d):i]
        if len(chunk_30) >= 50 and chunk_30["pred_return_4h"].std() > 1e-10:
            ic, _ = spearmanr(chunk_30["pred_return_4h"], chunk_30["actual_4h"])
            ic_30d.append(round(float(ic), 3) if not np.isnan(ic) else None)
        else:
            ic_30d.append(None)

    if not labels:
        return '<div style="color:#666">數據不足以計算 rolling IC</div>'

    # Zero line as a dataset with borderDash
    zero_line = [0] * len(labels)

    return f"""
    <div style="position:relative;height:180px">
      <canvas id="rollingIcChart"></canvas>
    </div>
    <script>
    (function(){{
      new Chart(document.getElementById('rollingIcChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [
            {{ label: '7d IC', data: {_json.dumps(ic_7d)},
               borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.05)',
               tension: 0.3, borderWidth: 2, pointRadius: 1, spanGaps: true }},
            {{ label: '30d IC', data: {_json.dumps(ic_30d)},
               borderColor: '#ff9800', backgroundColor: 'rgba(255,152,0,0.05)',
               tension: 0.3, borderWidth: 2, pointRadius: 1, spanGaps: true }},
            {{ label: 'Zero', data: {_json.dumps(zero_line)},
               borderColor: '#666', borderWidth: 1, borderDash: [4,4],
               pointRadius: 0, fill: false }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }},
                       filter: function(item) {{ return item.text !== 'Zero'; }} }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'Spearman IC', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 4. Pred vs Actual Scatter ──────────────────────────────────────────

def _build_scatter() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, pred_return_4h, strength_code
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                  AND pred_return_4h IS NOT NULL
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if len(rows) < 10:
        return '<div style="color:#666">數據不足</div>'

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    df["pred_return_4h"] = df["pred_return_4h"].astype(float)
    df["strength_code"] = df["strength_code"].fillna(1).astype(int)
    df["actual_4h"] = df["close"].shift(-4) / df["close"] - 1
    df = df.dropna(subset=["actual_4h"])

    if len(df) < 5:
        return '<div style="color:#666">等待 4h 回填中</div>'

    # Split by strength for coloring
    strong_pts = [{"x": round(float(r.pred_return_4h) * 100, 3),
                   "y": round(float(r.actual_4h) * 100, 3)}
                  for r in df[df["strength_code"] == 3].itertuples()]
    mod_pts = [{"x": round(float(r.pred_return_4h) * 100, 3),
                "y": round(float(r.actual_4h) * 100, 3)}
               for r in df[df["strength_code"] == 2].itertuples()]
    weak_pts = [{"x": round(float(r.pred_return_4h) * 100, 3),
                 "y": round(float(r.actual_4h) * 100, 3)}
                for r in df[df["strength_code"] <= 1].itertuples()]

    # Diagonal reference line (y=x)
    all_vals = [p["x"] for p in strong_pts + mod_pts + weak_pts] + \
               [p["y"] for p in strong_pts + mod_pts + weak_pts]
    if all_vals:
        lo = min(all_vals)
        hi = max(all_vals)
    else:
        lo, hi = -1, 1
    diag = [{"x": round(lo, 3), "y": round(lo, 3)},
            {"x": round(hi, 3), "y": round(hi, 3)}]

    return f"""
    <div style="position:relative;height:200px">
      <canvas id="predScatter"></canvas>
    </div>
    <div style="color:#8b949e;font-size:10px;margin-top:4px">
      X=預測 4h return %, Y=實際 4h return % | 虛線=完美預測
    </div>
    <script>
    (function(){{
      new Chart(document.getElementById('predScatter').getContext('2d'), {{
        type: 'scatter',
        data: {{
          datasets: [
            {{ label: 'Strong', data: {_json.dumps(strong_pts)},
               backgroundColor: 'rgba(244,67,54,0.7)', pointRadius: 5 }},
            {{ label: 'Moderate', data: {_json.dumps(mod_pts)},
               backgroundColor: 'rgba(255,152,0,0.7)', pointRadius: 4 }},
            {{ label: 'Weak', data: {_json.dumps(weak_pts)},
               backgroundColor: 'rgba(139,148,158,0.4)', pointRadius: 3 }},
            {{ label: 'y=x', data: {_json.dumps(diag)},
               type: 'line', borderColor: '#666', borderWidth: 1,
               borderDash: [4,4], pointRadius: 0, fill: false }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'Pred ret %', color: '#8b949e' }} }},
            y: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'Actual ret %', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 5. Feature Radar ───────────────────────────────────────────────────

def _build_feature_radar(state: dict, engine) -> str:
    """6-axis radar from latest indicator_history row with z-scores."""
    # Define feature groups with column mappings
    groups = {
        "Momentum": ["close"],
        "OrderFlow": ["bull_bear_power"],
        "Sentiment": ["dir_prob_up"],
        "Volatility": ["mag_pred"],
        "OI": ["cg_oi_close"],
        "Funding": ["cg_funding_close"],
    }

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Get available columns
            cur.execute("SHOW COLUMNS FROM indicator_history")
            existing_cols = {r["Field"] for r in cur.fetchall()}

            # Get latest row
            cur.execute("""
                SELECT * FROM indicator_history
                ORDER BY dt DESC LIMIT 1
            """)
            latest = cur.fetchone()

            # Get 7d stats for z-score
            stats: dict[str, tuple[float, float]] = {}
            for group, cols in groups.items():
                for col in cols:
                    if col in existing_cols:
                        cur.execute(f"""
                            SELECT AVG(`{col}`) as m, STDDEV(`{col}`) as s
                            FROM indicator_history
                            WHERE dt >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                              AND `{col}` IS NOT NULL
                        """)
                        r = cur.fetchone()
                        if r and r["m"] is not None and r["s"] and float(r["s"]) > 1e-10:
                            stats[col] = (float(r["m"]), float(r["s"]))
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if not latest:
        return '<div style="color:#666">無數據</div>'

    # Compute z-scores per group (use first available col in each group)
    radar_labels: list[str] = []
    z_scores: list[float] = []

    for group, cols in groups.items():
        z = 0.0
        found = False
        for col in cols:
            if col in existing_cols and latest.get(col) is not None and col in stats:
                mean, std = stats[col]
                z = (float(latest[col]) - mean) / std
                z = max(-3.0, min(3.0, z))  # clamp
                found = True
                break
        radar_labels.append(group)
        z_scores.append(round(z, 2) if found else 0.0)

    # Normalize to 0-100 for radar display (z=-3 -> 0, z=0 -> 50, z=+3 -> 100)
    radar_values = [round((z + 3) / 6 * 100, 1) for z in z_scores]

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      6 維特徵群 Z-Score (7d 基準) | 50=均值, 0=極低, 100=極高
    </div>
    <div style="position:relative;height:200px">
      <canvas id="featRadar"></canvas>
    </div>
    <script>
    (function(){{
      new Chart(document.getElementById('featRadar').getContext('2d'), {{
        type: 'radar',
        data: {{
          labels: {_json.dumps(radar_labels)},
          datasets: [{{
            label: 'Z-Score',
            data: {_json.dumps(radar_values)},
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.15)',
            borderWidth: 2,
            pointBackgroundColor: '#58a6ff',
            pointRadius: 3
          }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }}
          }},
          scales: {{
            r: {{
              min: 0, max: 100,
              ticks: {{ color: '#8b949e', font: {{ size: 8 }}, backdropColor: 'transparent',
                        stepSize: 25 }},
              grid: {{ color: '#30363d' }},
              angleLines: {{ color: '#30363d' }},
              pointLabels: {{ color: '#c9d1d9', font: {{ size: 11 }} }}
            }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 6. Regime + Signal Overlay ─────────────────────────────────────────

def _build_regime_signals() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, regime_code, close
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                ORDER BY dt ASC
            """)
            hist_rows = cur.fetchall()

            cur.execute("""
                SELECT signal_time, direction, strength, correct, filled
                FROM tracked_signals
                WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                ORDER BY signal_time ASC
            """)
            sig_rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if not hist_rows:
        return '<div style="color:#666">數據不足</div>'

    df = pd.DataFrame(hist_rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    regime_colors_map = {2: "rgba(76,175,80,0.3)", -2: "rgba(244,67,54,0.3)",
                         0: "rgba(255,152,0,0.3)", -99: "rgba(102,102,102,0.3)"}

    labels = [dt.strftime("%m/%d %H:%M") for dt in df["dt"]]
    # Bar heights = 1 for all, colored by regime
    bar_colors = [regime_colors_map.get(int(r["regime_code"] or 0), "rgba(102,102,102,0.3)")
                  for _, r in df.iterrows()]
    bar_data = [1] * len(df)

    # Build signal scatter overlay
    # Map signal times to nearest index in labels
    dt_list = df["dt"].tolist()
    sig_points: list[dict] = []
    for s in sig_rows:
        t = s["signal_time"]
        if not hasattr(t, "timestamp"):
            continue
        t_ts = pd.Timestamp(t)
        # Find nearest index
        diffs = [abs((d - t_ts).total_seconds()) for d in dt_list]
        if not diffs:
            continue
        idx = int(np.argmin(diffs))

        is_up = s["direction"] == "UP"
        is_strong = s["strength"] == "Strong"

        sig_points.append({
            "x": idx,
            "y": 0.5,
            "r": 8 if is_strong else 5,
            "bg": "#4caf50" if is_up else "#f44336",
        })

    up_pts = [{"x": p["x"], "y": p["y"], "r": p["r"]}
              for p in sig_points if p["bg"] == "#4caf50"]
    dn_pts = [{"x": p["x"], "y": p["y"], "r": p["r"]}
              for p in sig_points if p["bg"] == "#f44336"]

    regime_names = {2: "BULL", -2: "BEAR", 0: "CHOPPY", -99: "WARMUP"}
    # Count regime hours
    regime_counts: dict[str, int] = {}
    for _, r in df.iterrows():
        name = regime_names.get(int(r["regime_code"] or 0), "?")
        regime_counts[name] = regime_counts.get(name, 0) + 1
    summary = " | ".join(f"{k}: {v}h" for k, v in
                         sorted(regime_counts.items(), key=lambda x: -x[1]))

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      背景=Regime (綠=BULL, 紅=BEAR, 橘=CHOPPY) | 圓點=信號 (綠=UP, 紅=DOWN, 大=Strong)
      <br>{summary}
    </div>
    <div style="position:relative;height:180px">
      <canvas id="regimeSignals"></canvas>
    </div>
    <script>
    (function(){{
      var barColors = {_json.dumps(bar_colors)};
      new Chart(document.getElementById('regimeSignals').getContext('2d'), {{
        type: 'bar',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [
            {{ label: 'Regime', data: {_json.dumps(bar_data)},
               backgroundColor: barColors, borderWidth: 0, barPercentage: 1.0,
               categoryPercentage: 1.0, order: 2 }},
            {{ label: 'UP Signal', type: 'bubble',
               data: {_json.dumps(up_pts)},
               backgroundColor: 'rgba(76,175,80,0.9)', order: 1 }},
            {{ label: 'DOWN Signal', type: 'bubble',
               data: {_json.dumps(dn_pts)},
               backgroundColor: 'rgba(244,67,54,0.9)', order: 1 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 8 }}, maxRotation: 45,
                            maxTicksLimit: 20 }},
                  grid: {{ display: false }} }},
            y: {{ display: false, min: 0, max: 1.2 }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 7. Real-time Gauges ────────────────────────────────────────────────

def _build_gauges() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM indicator_history
                ORDER BY dt DESC LIMIT 1
            """)
            latest = cur.fetchone()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if not latest:
        return '<div style="color:#666">無數據</div>'

    def _safe(col, default=0.0):
        v = latest.get(col)
        return float(v) if v is not None else default

    funding = _safe("cg_funding_close")
    oi_close = _safe("cg_oi_close")
    ls_ratio = _safe("cg_ls_ratio", 1.0)
    bbp = _safe("bull_bear_power")
    confidence = _safe("confidence_score")
    dvol = _safe("deribit_dvol") if "deribit_dvol" in (latest or {}) else None

    # Build circular gauge HTML
    def _gauge(label: str, value: float, display: str, min_v: float, max_v: float,
               color: str, unit: str = "") -> str:
        # Clamp
        clamped = max(min_v, min(value, max_v))
        pct = (clamped - min_v) / (max_v - min_v) * 100 if max_v > min_v else 50
        # SVG arc gauge
        angle = pct / 100 * 180  # 0-180 degrees
        rad = angle * 3.14159 / 180
        # Path for arc
        r = 40
        cx, cy = 50, 55
        x1 = cx - r  # start at left
        y1 = cy
        x2 = cx - r * np.cos(rad)
        y2 = cy - r * np.sin(rad)
        large_arc = 1 if angle > 90 else 0

        return f"""<div style="text-align:center;padding:4px">
          <svg width="100" height="65" viewBox="0 0 100 65">
            <path d="M {cx-r},{cy} A {r},{r} 0 0 1 {cx+r},{cy}"
                  fill="none" stroke="#30363d" stroke-width="6" stroke-linecap="round"/>
            <path d="M {cx-r},{cy} A {r},{r} 0 {large_arc} 1 {x2:.1f},{y2:.1f}"
                  fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round"/>
            <text x="{cx}" y="{cy-8}" text-anchor="middle" fill="#c9d1d9"
                  font-size="12" font-weight="600">{display}</text>
            <text x="{cx}" y="{cy+2}" text-anchor="middle" fill="#8b949e"
                  font-size="8">{unit}</text>
          </svg>
          <div style="color:#8b949e;font-size:10px;margin-top:-4px">{label}</div>
        </div>"""

    # Color logic
    funding_color = "#ff9800" if abs(funding) > 0.0001 else "#4caf50"
    bbp_color = "#4caf50" if bbp > 0.1 else "#f44336" if bbp < -0.1 else "#8b949e"
    conf_color = "#4caf50" if confidence >= 75 else "#ff9800" if confidence >= 60 else "#8b949e"
    ls_color = "#4caf50" if ls_ratio > 1.05 else "#f44336" if ls_ratio < 0.95 else "#8b949e"
    dvol_color = "#ff9800" if (dvol or 0) > 60 else "#4caf50"

    gauges = [
        _gauge("Funding Rate", funding, f"{funding*100:.4f}%",
               -0.001, 0.001, funding_color, "8h rate"),
        _gauge("OI", oi_close, f"{oi_close/1e9:.1f}B" if oi_close > 1e6 else "N/A",
               0, oi_close * 1.5 if oi_close > 0 else 1, "#e040fb", "USD"),
        _gauge("L/S Ratio", ls_ratio, f"{ls_ratio:.3f}",
               0.8, 1.2, ls_color, "Long/Short"),
        _gauge("BBP", bbp, f"{bbp:.3f}",
               -1.0, 1.0, bbp_color, "Bull Bear Power"),
        _gauge("Confidence", confidence, f"{confidence:.0f}",
               0, 100, conf_color, "Score"),
        _gauge("DVOL", dvol or 0, f"{dvol:.1f}" if dvol else "N/A",
               20, 100, dvol_color, "Deribit Vol"),
    ]

    return f"""
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:4px">
      {''.join(gauges)}
    </div>"""


# ── 8. Signal on Price Mini Chart ──────────────────────────────────────

def _build_signal_price_chart() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 72 HOUR)
                ORDER BY dt ASC
            """)
            price_rows = cur.fetchall()

            cur.execute("""
                SELECT signal_time, direction, strength, correct, filled
                FROM tracked_signals
                WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 72 HOUR)
                ORDER BY signal_time ASC
            """)
            sig_rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if not price_rows:
        return '<div style="color:#666">數據不足</div>'

    df = pd.DataFrame(price_rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    labels = []
    for dt in df["dt"]:
        if hasattr(dt, "replace"):
            dt_local = dt.replace(tzinfo=timezone.utc).astimezone(TZ8)
            labels.append(dt_local.strftime("%m/%d %H:%M"))
        else:
            labels.append(str(dt)[:16])

    prices = [round(float(r), 0) for r in df["close"]]

    # Map signals to chart positions
    dt_list = df["dt"].tolist()
    up_strong: list[dict] = []
    up_mod: list[dict] = []
    dn_strong: list[dict] = []
    dn_mod: list[dict] = []

    for s in sig_rows:
        t = s["signal_time"]
        if not hasattr(t, "timestamp"):
            continue
        t_ts = pd.Timestamp(t)
        diffs = [abs((d - t_ts).total_seconds()) for d in dt_list]
        if not diffs:
            continue
        idx = int(np.argmin(diffs))
        price_at = prices[idx] if idx < len(prices) else prices[-1]

        pt = {"x": idx, "y": price_at}
        is_up = s["direction"] == "UP"
        is_strong = s["strength"] == "Strong"

        if is_up and is_strong:
            up_strong.append(pt)
        elif is_up:
            up_mod.append(pt)
        elif is_strong:
            dn_strong.append(pt)
        else:
            dn_mod.append(pt)

    return f"""
    <div style="position:relative;height:200px">
      <canvas id="sigPriceChart"></canvas>
    </div>
    <div style="color:#8b949e;font-size:10px;margin-top:4px">
      綠三角=UP, 紅三角=DOWN | 大=Strong, 小=Moderate
    </div>
    <script>
    (function(){{
      new Chart(document.getElementById('sigPriceChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [
            {{ label: 'BTC Price', data: {_json.dumps(prices)},
               borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.05)',
               fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0, order: 2 }},
            {{ label: 'UP Strong', type: 'scatter',
               data: {_json.dumps(up_strong)},
               backgroundColor: '#4caf50', pointRadius: 8,
               pointStyle: 'triangle', order: 1 }},
            {{ label: 'UP Moderate', type: 'scatter',
               data: {_json.dumps(up_mod)},
               backgroundColor: '#4caf50', pointRadius: 5,
               pointStyle: 'triangle', order: 1 }},
            {{ label: 'DOWN Strong', type: 'scatter',
               data: {_json.dumps(dn_strong)},
               backgroundColor: '#f44336', pointRadius: 8,
               pointStyle: 'triangle', pointRotation: 180, order: 1 }},
            {{ label: 'DOWN Moderate', type: 'scatter',
               data: {_json.dumps(dn_mod)},
               backgroundColor: '#f44336', pointRadius: 5,
               pointStyle: 'triangle', pointRotation: 180, order: 1 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45,
                            maxTicksLimit: 15 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'BTC Price', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""
