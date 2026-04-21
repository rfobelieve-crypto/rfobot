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
        section("Signal on Price (72h)", "sigPrice", True,
                _build_signal_price_chart()),
        section("Real-time Gauges", "rtGauges", True, _build_gauges()),
        section("Equity Curve by Tier", "eqTier", True, _build_equity_by_tier()),
        section("Rolling IC (7d / 30d)", "rollingIc", True, _build_rolling_ic()),
        section("Pred vs Actual Scatter (7d)", "predScatter", True, _build_scatter()),
        section("Feature Radar (Z-Score)", "featRadar", True,
                _build_feature_radar(state, engine)),
        section("Regime Timeline (7d)", "regimeSig", True,
                _build_regime_signals()),
        section("Signal Heatmap (30d)", "sigHeatmap", True, _build_signal_heatmap()),
    ]
    return "\n".join(parts)


# ── 1. Signal on Price (72h) ──────────────────────────────────────────

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
        dt_local = dt.tz_localize("UTC").tz_convert("Asia/Taipei") if dt.tzinfo is None else dt.astimezone(TZ8)
        labels.append(dt_local.strftime("%m/%d %H:%M"))

    prices = [round(float(r), 0) for r in df["close"]]

    # Map signals onto price line — use label index matching
    dt_list = df["dt"].tolist()
    up_strong, up_mod, dn_strong, dn_mod = [], [], [], []

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

        # Use the label string as x, not a numeric index
        pt = {"x": labels[idx], "y": price_at}
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
    <div style="position:relative;height:220px;overflow:hidden">
      <canvas id="sigPriceChart"></canvas>
    </div>
    <div style="color:#8b949e;font-size:10px;margin-top:4px">
      綠=UP, 紅=DOWN | 大點=Strong, 小點=Moderate
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
            {{ label: 'UP Mod', type: 'scatter',
               data: {_json.dumps(up_mod)},
               backgroundColor: '#66bb6a', pointRadius: 5,
               pointStyle: 'triangle', order: 1 }},
            {{ label: 'DN Strong', type: 'scatter',
               data: {_json.dumps(dn_strong)},
               backgroundColor: '#f44336', pointRadius: 8,
               pointStyle: 'rectRot', order: 1 }},
            {{ label: 'DN Mod', type: 'scatter',
               data: {_json.dumps(dn_mod)},
               backgroundColor: '#ef5350', pointRadius: 5,
               pointStyle: 'rectRot', order: 1 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 8 }}, maxRotation: 45,
                            maxTicksLimit: 12, autoSkip: true }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 2. Real-time Gauges ───────────────────────────────────────────────

def _build_gauges() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM indicator_history ORDER BY dt DESC LIMIT 1")
            latest = cur.fetchone()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if not latest:
        return '<div style="color:#666">無數據</div>'

    def _safe(col, default=0.0):
        v = latest.get(col)
        return float(v) if v is not None else default

    bbp = _safe("bull_bear_power")
    confidence = _safe("confidence_score")
    dir_prob = _safe("dir_prob_up", 0.5)
    mag = _safe("mag_pred")
    pred_ret = _safe("pred_return_4h")

    # Use simple horizontal gauge bars instead of SVG arcs (more reliable layout)
    def _bar(label: str, value: float, display: str, pct: float, color: str) -> str:
        pct = max(0, min(100, pct))
        return f"""<div style="margin:6px 0">
          <div style="display:flex;justify-content:space-between;margin-bottom:2px">
            <span style="color:#8b949e;font-size:11px">{label}</span>
            <span style="color:{color};font-size:12px;font-weight:600">{display}</span>
          </div>
          <div style="background:#21262d;border-radius:4px;height:8px;overflow:hidden">
            <div style="background:{color};height:100%;width:{pct:.0f}%;border-radius:4px;
                         transition:width 0.3s"></div>
          </div>
        </div>"""

    bbp_pct = (bbp + 1) / 2 * 100  # -1..+1 -> 0..100
    bbp_color = "#4caf50" if bbp > 0.1 else "#f44336" if bbp < -0.1 else "#8b949e"

    conf_color = "#4caf50" if confidence >= 75 else "#ff9800" if confidence >= 60 else "#8b949e"

    dir_pct = dir_prob * 100
    dir_color = "#4caf50" if dir_prob > 0.55 else "#f44336" if dir_prob < 0.45 else "#8b949e"

    mag_pct = min(mag / 0.01 * 100, 100) if mag else 0  # 1% = full
    mag_color = "#ff9800" if mag > 0.005 else "#58a6ff"

    ret_display = f"{pred_ret*100:+.3f}%"
    ret_pct = min(abs(pred_ret) / 0.005 * 50 + 50, 100)  # center at 50
    ret_color = "#4caf50" if pred_ret > 0 else "#f44336" if pred_ret < 0 else "#8b949e"

    bars = [
        _bar("BBP (Bull Bear Power)", bbp, f"{bbp:+.3f}", bbp_pct, bbp_color),
        _bar("Confidence", confidence, f"{confidence:.0f}", confidence, conf_color),
        _bar("P(UP)", dir_prob, f"{dir_prob:.1%}", dir_pct, dir_color),
        _bar("Magnitude", mag, f"{mag*100:.3f}%", mag_pct, mag_color),
        _bar("Pred Return 4h", pred_ret, ret_display, ret_pct, ret_color),
    ]

    return f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px">
      <div>{''.join(bars[:3])}</div>
      <div>{''.join(bars[3:])}</div>
    </div>"""


# ── 3. Equity Curve by Tier ──────────────────────────────────────────

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

    if len(rows) < 2:
        return '<div style="color:#666">信號不足</div>'

    # Build unified timeline
    combined = []
    for r in rows:
        ret = float(r["actual_return_4h"] or 0)
        if r["direction"] == "DOWN":
            ret = -ret
        t = r["signal_time"]
        lbl = t.strftime("%m/%d %H:%M") if hasattr(t, "strftime") else str(t)[:16]
        combined.append((lbl, r["strength"], ret * 100))

    combined.sort(key=lambda x: x[0])

    labels, s_vals, m_vals = [], [], []
    cum_s, cum_m = 0.0, 0.0
    n_s, n_m = 0, 0

    for lbl, tier, ret in combined:
        labels.append(lbl)
        if tier == "Strong":
            cum_s += ret
            n_s += 1
        elif tier == "Moderate":
            cum_m += ret
            n_m += 1
        s_vals.append(round(cum_s, 2))
        m_vals.append(round(cum_m, 2))

    return f"""
    <div class="grid grid-3" style="margin-bottom:8px">
      {card("Strong", f'{cum_s:+.2f}%', f'{n_s} 筆',
            "#4caf50" if cum_s >= 0 else "#f44336")}
      {card("Moderate", f'{cum_m:+.2f}%', f'{n_m} 筆',
            "#4caf50" if cum_m >= 0 else "#f44336")}
      {card("總計", str(len(rows)), "")}
    </div>
    <div style="position:relative;height:180px;overflow:hidden">
      <canvas id="equityTier"></canvas>
    </div>
    <script>
    (function(){{
      new Chart(document.getElementById('equityTier').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [
            {{ label: 'Strong', data: {_json.dumps(s_vals)},
               borderColor: '#f44336', fill: false, tension: 0.3,
               borderWidth: 2, pointRadius: 2 }},
            {{ label: 'Moderate', data: {_json.dumps(m_vals)},
               borderColor: '#ff9800', fill: false, tension: 0.3,
               borderWidth: 2, pointRadius: 2 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 8 }}, maxRotation: 45,
                            maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: '累計 %', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 4. Rolling IC Chart ──────────────────────────────────────────────

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

    window_7d = min(7 * 24, len(df) // 2)
    labels, ic_7d, ic_30d = [], [], []
    step = max(1, len(df) // 60)

    for i in range(window_7d, len(df), step):
        labels.append(df["dt"].iloc[i].strftime("%m/%d"))

        chunk_7 = df.iloc[max(0, i - window_7d):i]
        if len(chunk_7) >= 20 and chunk_7["pred_return_4h"].std() > 1e-10:
            ic, _ = spearmanr(chunk_7["pred_return_4h"], chunk_7["actual_4h"])
            ic_7d.append(round(float(ic), 3) if not np.isnan(ic) else None)
        else:
            ic_7d.append(None)

        window_30d = min(30 * 24, i)
        chunk_30 = df.iloc[max(0, i - window_30d):i]
        if len(chunk_30) >= 50 and chunk_30["pred_return_4h"].std() > 1e-10:
            ic, _ = spearmanr(chunk_30["pred_return_4h"], chunk_30["actual_4h"])
            ic_30d.append(round(float(ic), 3) if not np.isnan(ic) else None)
        else:
            ic_30d.append(None)

    if not labels:
        return '<div style="color:#666">數據不足以計算 rolling IC</div>'

    zero_line = [0] * len(labels)

    return f"""
    <div style="position:relative;height:180px;overflow:hidden">
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
               borderColor: '#58a6ff', tension: 0.3, borderWidth: 2,
               pointRadius: 1, spanGaps: true }},
            {{ label: '30d IC', data: {_json.dumps(ic_30d)},
               borderColor: '#ff9800', tension: 0.3, borderWidth: 2,
               pointRadius: 1, spanGaps: true }},
            {{ label: '', data: {_json.dumps(zero_line)},
               borderColor: '#444', borderWidth: 1, borderDash: [4,4],
               pointRadius: 0, fill: false }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45,
                            maxTicksLimit: 15 }}, grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'Spearman IC', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 5. Pred vs Actual Scatter ────────────────────────────────────────

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

    strong_pts = [{"x": round(float(r.pred_return_4h) * 100, 3),
                   "y": round(float(r.actual_4h) * 100, 3)}
                  for r in df[df["strength_code"] == 3].itertuples()]
    mod_pts = [{"x": round(float(r.pred_return_4h) * 100, 3),
                "y": round(float(r.actual_4h) * 100, 3)}
               for r in df[df["strength_code"] == 2].itertuples()]
    weak_pts = [{"x": round(float(r.pred_return_4h) * 100, 3),
                 "y": round(float(r.actual_4h) * 100, 3)}
                for r in df[df["strength_code"] <= 1].itertuples()]

    all_vals = ([p["x"] for p in strong_pts + mod_pts + weak_pts] +
                [p["y"] for p in strong_pts + mod_pts + weak_pts])
    lo = min(all_vals) if all_vals else -1
    hi = max(all_vals) if all_vals else 1
    diag = [{"x": round(lo, 3), "y": round(lo, 3)},
            {"x": round(hi, 3), "y": round(hi, 3)}]

    return f"""
    <div style="position:relative;height:200px;overflow:hidden">
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
            {{ label: 'y=x', type: 'line', data: {_json.dumps(diag)},
               borderColor: '#666', borderWidth: 1, borderDash: [4,4],
               pointRadius: 0, fill: false }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'Pred %', color: '#8b949e' }} }},
            y: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'Actual %', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 6. Feature Radar ─────────────────────────────────────────────────

def _build_feature_radar(state: dict, engine) -> str:
    groups = {
        "Momentum": ["close"],
        "OrderFlow": ["bull_bear_power"],
        "Sentiment": ["dir_prob_up"],
        "Volatility": ["mag_pred"],
    }

    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM indicator_history")
            existing_cols = {r["Field"] for r in cur.fetchall()}

            cur.execute("SELECT * FROM indicator_history ORDER BY dt DESC LIMIT 1")
            latest = cur.fetchone()

            stats = {}
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

    radar_labels, z_scores = [], []
    for group, cols in groups.items():
        z = 0.0
        for col in cols:
            if col in existing_cols and latest.get(col) is not None and col in stats:
                mean, std = stats[col]
                z = (float(latest[col]) - mean) / std
                z = max(-3.0, min(3.0, z))
                break
        radar_labels.append(group)
        z_scores.append(round(z, 2))

    radar_values = [round((z + 3) / 6 * 100, 1) for z in z_scores]

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      4 維特徵群 Z-Score (7d 基準) | 50=均值, 0=極低, 100=極高
    </div>
    <div style="position:relative;height:240px;max-width:350px;margin:0 auto;overflow:hidden">
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
            pointRadius: 4
          }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            r: {{
              min: 0, max: 100,
              ticks: {{ color: '#8b949e', font: {{ size: 9 }}, backdropColor: 'transparent',
                        stepSize: 25 }},
              grid: {{ color: '#30363d' }},
              angleLines: {{ color: '#30363d' }},
              pointLabels: {{ color: '#c9d1d9', font: {{ size: 12 }} }}
            }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── 7. Regime Timeline ───────────────────────────────────────────────

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
                SELECT signal_time, direction, strength
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

    regime_colors_map = {
        2: "#4caf50", -2: "#f44336", 0: "#ff9800", -99: "#666"
    }
    regime_names = {2: "BULL", -2: "BEAR", 0: "CHOPPY", -99: "WARMUP"}

    # Build HTML regime bar + signal markers
    n = len(hist_rows)
    regime_counts = {}

    bar_cells = []
    for r in hist_rows:
        code = int(r["regime_code"] or 0)
        color = regime_colors_map.get(code, "#666")
        name = regime_names.get(code, "?")
        regime_counts[name] = regime_counts.get(name, 0) + 1
        bar_cells.append(f'<div style="flex:1;background:{color};min-width:1px" '
                         f'title="{name}"></div>')

    summary = " | ".join(f"{k}: {v}h" for k, v in
                         sorted(regime_counts.items(), key=lambda x: -x[1]))

    # Signal markers positioned relative to timeline
    dt_list = [r["dt"] for r in hist_rows]
    sig_markers = []
    for s in sig_rows:
        t = s["signal_time"]
        if not hasattr(t, "timestamp"):
            continue
        t_ts = pd.Timestamp(t)
        diffs = [abs((pd.Timestamp(d) - t_ts).total_seconds()) for d in dt_list]
        if not diffs:
            continue
        idx = int(np.argmin(diffs))
        left_pct = idx / max(n - 1, 1) * 100

        is_up = s["direction"] == "UP"
        is_strong = s["strength"] == "Strong"
        color = "#4caf50" if is_up else "#f44336"
        size = 10 if is_strong else 6
        symbol = "&#9650;" if is_up else "&#9660;"

        sig_markers.append(
            f'<div style="position:absolute;left:{left_pct:.1f}%;top:-2px;'
            f'color:{color};font-size:{size}px;transform:translateX(-50%)"'
            f' title="{s["direction"]} {s["strength"]}">{symbol}</div>')

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      {summary} | 三角形=信號 (綠=UP, 紅=DOWN, 大=Strong)
    </div>
    <div style="position:relative;margin:16px 0 8px">
      <div style="display:flex;height:20px;border-radius:4px;overflow:hidden;gap:1px">
        {''.join(bar_cells)}
      </div>
      {''.join(sig_markers)}
    </div>
    <div style="display:flex;gap:12px;margin-top:8px">
      <span style="font-size:10px"><span style="color:#4caf50">&#9632;</span> BULL</span>
      <span style="font-size:10px"><span style="color:#f44336">&#9632;</span> BEAR</span>
      <span style="font-size:10px"><span style="color:#ff9800">&#9632;</span> CHOPPY</span>
    </div>"""


# ── 8. Signal Heatmap (30d) ──────────────────────────────────────────

def _build_signal_heatmap() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT signal_time, direction, confidence, correct, filled
                FROM tracked_signals
                WHERE signal_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                ORDER BY signal_time ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">數據載入失敗: {e}</div>'

    if not rows:
        return '<div style="color:#666">無數據</div>'

    # Build HTML grid: rows = confidence buckets, columns = dates
    date_map = {}  # date_str -> [signals]
    for r in rows:
        conf = float(r["confidence"] or 0)
        if conf < 60:
            continue
        t = r["signal_time"]
        if hasattr(t, "replace"):
            t_local = t.replace(tzinfo=timezone.utc).astimezone(TZ8)
            date_str = t_local.strftime("%m/%d")
        else:
            date_str = str(t)[:10]

        filled = bool(r["filled"])
        correct = bool(r["correct"]) if filled else None

        if correct is True:
            bg = "#4caf50"
        elif correct is False:
            bg = "#f44336"
        else:
            bg = "#555"

        arrow = "&#9650;" if r["direction"] == "UP" else "&#9660;"
        d_color = "#4caf50" if r["direction"] == "UP" else "#f44336"

        if date_str not in date_map:
            date_map[date_str] = []
        date_map[date_str].append(
            f'<div style="display:inline-block;background:{bg};color:white;'
            f'padding:2px 6px;border-radius:4px;margin:2px;font-size:11px" '
            f'title="conf={conf:.0f} {r["direction"]}">'
            f'<span style="color:{d_color}">{arrow}</span> {conf:.0f}</div>')

    if not date_map:
        return '<div style="color:#666">無符合條件的信號</div>'

    # Render as date rows
    html_rows = []
    for date_str in sorted(date_map.keys()):
        signals = "".join(date_map[date_str])
        html_rows.append(
            f'<div style="display:flex;align-items:center;margin:3px 0">'
            f'<div style="width:50px;color:#8b949e;font-size:11px;flex-shrink:0">{date_str}</div>'
            f'<div style="flex:1">{signals}</div></div>')

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      綠底=正確, 紅底=錯誤, 灰底=待定 | 數字=信心分數
    </div>
    {''.join(html_rows)}"""
