"""Tab 3: Market Intelligence — what's driving the signal."""
from __future__ import annotations

import json as _json
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from indicator.dashboard_tabs._components import (
    card, section, get_db_conn, TZ8,
)

logger = logging.getLogger(__name__)


def render_market(state: dict, engine) -> str:
    parts = [
        section("Top-5 特徵值 + Z-Score", "top5feat", True,
                _build_top_features(state, engine)),
        section("Mag 預測 vs 實際波動率", "magvol", True, _build_mag_vs_realized()),
        section("Funding Rate 環境", "funding", True, _build_funding_env()),
        section("OI 變化 + 清算", "oi_liq", True, _build_oi_section()),
        section("多空比 + 大戶持倉", "ls_ratio", True, _build_ls_ratio()),
        section("跨時間框架一致性", "xtf", True, _build_cross_timeframe()),
    ]
    return "\n".join(parts)


# ── Top-5 Features ───────────────────────────────────────────────────

def _build_top_features(state: dict, engine) -> str:
    """Show the top-5 most important features with current values and z-scores."""
    try:
        importance_path = "indicator/model_artifacts/dual_model/direction_importance.csv"
        imp_df = pd.read_csv(importance_path)
        imp_df.columns = ["feature", "importance"] + list(imp_df.columns[2:])
        top5 = imp_df.nlargest(5, "importance")["feature"].tolist()
    except Exception:
        top5 = []

    if not top5:
        return '<div style="color:#666">無法載入特徵重要性</div>'

    # Try to get current feature values from engine's last prediction state
    pred = state.get("last_prediction", {})
    features = pred.get("features", {})

    # Get recent history for z-score computation
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            # Check which columns exist
            cur.execute("SHOW COLUMNS FROM indicator_history")
            existing_cols = {r["Field"] for r in cur.fetchall()}
        conn.close()
    except Exception:
        existing_cols = set()

    rows_html = []
    for feat in top5:
        val = features.get(feat)
        val_str = f"{val:.4f}" if val is not None else "N/A"

        # Importance
        imp_val = imp_df[imp_df["feature"] == feat]["importance"].values
        imp_str = f"{imp_val[0]:.1f}" if len(imp_val) > 0 else "?"

        # Z-score placeholder — compute from indicator_history if column exists
        z_str = "N/A"
        z_color = "#8b949e"

        if feat in existing_cols and val is not None:
            try:
                conn = get_db_conn()
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT AVG(`{feat}`) as m, STDDEV(`{feat}`) as s
                        FROM indicator_history
                        WHERE dt >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                          AND `{feat}` IS NOT NULL
                    """)
                    r = cur.fetchone()
                conn.close()
                if r and r["m"] is not None and r["s"] and float(r["s"]) > 1e-10:
                    z = (val - float(r["m"])) / float(r["s"])
                    z_str = f"{z:+.2f}"
                    z_color = "#f44336" if abs(z) > 2 else "#ff9800" if abs(z) > 1 else "#4caf50"
            except Exception:
                pass

        # Bar visualization for importance
        bar_w = min(float(imp_str) / 5 * 100 if imp_str != "?" else 0, 120)

        rows_html.append(
            f"<tr>"
            f"<td style='font-size:11px;max-width:200px;overflow:hidden;text-overflow:ellipsis'>"
            f"<code>{feat}</code></td>"
            f"<td style='text-align:right;font-family:monospace'>{val_str}</td>"
            f"<td style='text-align:right;color:{z_color};font-family:monospace'>{z_str}</td>"
            f"<td><div style='background:#58a6ff;height:8px;width:{bar_w}px;"
            f"border-radius:3px;opacity:0.7'></div></td>"
            f"<td style='text-align:right;color:#8b949e;font-size:10px'>{imp_str}</td>"
            f"</tr>"
        )

    return f"""
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      Direction Model 前 5 重要特徵 — 當前值 + 7 天 Z-Score
    </div>
    <table>
      <tr><th>特徵</th><th>當前值</th><th>Z-Score</th><th colspan="2">重要性</th></tr>
      {''.join(rows_html)}
    </table>"""


# ── Mag vs Realized Volatility ───────────────────────────────────────

def _build_mag_vs_realized() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, mag_pred
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 72 HOUR)
                  AND mag_pred IS NOT NULL
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">{e}</div>'

    if len(rows) < 10:
        return '<div style="color:#666">數據不足</div>'

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["realized_4h"] = abs(df["close"].shift(-4) / df["close"] - 1)
    df["mag_pred"] = df["mag_pred"].astype(float)
    df = df.dropna(subset=["realized_4h"])

    if len(df) < 5:
        return '<div style="color:#666">等待 4h 回填中</div>'

    labels = [dt.strftime("%m/%d %H:%M") for dt in df["dt"]]
    mag_preds = [round(float(v) * 100, 3) for v in df["mag_pred"]]
    realized = [round(float(v) * 100, 3) for v in df["realized_4h"]]

    # Summary stats
    pred_mean = np.mean(mag_preds)
    real_mean = np.mean(realized)
    ratio = pred_mean / real_mean if real_mean > 0 else 0

    return f"""
    <div class="grid grid-3" style="margin-bottom:8px">
      {card("預測均值", f'{pred_mean:.3f}%', "")}
      {card("實際均值", f'{real_mean:.3f}%', "")}
      {card("預測/實際", f'{ratio:.2f}x',
            "高估" if ratio > 1.3 else "低估" if ratio < 0.7 else "校準良好",
            "#ff9800" if abs(ratio - 1) > 0.3 else "#4caf50")}
    </div>
    <div style="position:relative;height:180px">
      <canvas id="magVolChart"></canvas>
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('magVolChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [
            {{ label: 'Mag 預測 %', data: {_json.dumps(mag_preds)},
               borderColor: '#58a6ff', borderWidth: 2, pointRadius: 1, tension: 0.3 }},
            {{ label: '實際 |ret| %', data: {_json.dumps(realized)},
               borderColor: '#ff9800', borderWidth: 2, pointRadius: 1, tension: 0.3 }}
          ]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: '|return| %', color: '#8b949e' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Funding Rate Environment ─────────────────────────────────────────

def _build_funding_env() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, cg_funding_close
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 72 HOUR)
                  AND cg_funding_close IS NOT NULL
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">{e}</div>'

    if not rows:
        return '<div style="color:#666">無 Funding 數據</div>'

    df = pd.DataFrame(rows)
    df["funding_pct"] = df["cg_funding_close"].astype(float) * 100

    current = df["funding_pct"].iloc[-1]
    avg_8h = df["funding_pct"].tail(8).mean()
    avg_24h = df["funding_pct"].tail(24).mean()
    pctile = (df["funding_pct"] <= current).mean() * 100

    # Determine sentiment
    if current > 0.02:
        sentiment = "極度做多"
        sent_color = "#f44336"
    elif current > 0.005:
        sentiment = "偏多"
        sent_color = "#ff9800"
    elif current < -0.02:
        sentiment = "極度做空"
        sent_color = "#4caf50"
    elif current < -0.005:
        sentiment = "偏空"
        sent_color = "#58a6ff"
    else:
        sentiment = "中性"
        sent_color = "#8b949e"

    labels = [r["dt"].strftime("%m/%d %H:%M") if hasattr(r["dt"], "strftime")
              else str(r["dt"])[:16] for _, r in df.iterrows()]
    values = df["funding_pct"].round(4).tolist()

    return f"""
    <div class="grid grid-4">
      {card("當前 Funding", f'{current:.4f}%', sentiment, sent_color)}
      {card("8h 均值", f'{avg_8h:.4f}%', "")}
      {card("24h 均值", f'{avg_24h:.4f}%', "")}
      {card("歷史百分位", f'{pctile:.0f}%', "72h 窗口")}
    </div>
    <div style="position:relative;height:150px">
      <canvas id="fundingChart"></canvas>
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('fundingChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [{{ data: {_json.dumps(values)},
            borderColor: '#ff9800', backgroundColor: 'rgba(255,152,0,0.05)',
            fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0 }}]
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
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#ff9800', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'Funding %', color: '#ff9800' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── OI Changes ───────────────────────────────────────────────────────

def _build_oi_section() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, cg_oi_close, close
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 72 HOUR)
                  AND cg_oi_close IS NOT NULL
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">{e}</div>'

    if len(rows) < 5:
        return '<div style="color:#666">數據不足</div>'

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])
    df["oi"] = df["cg_oi_close"].astype(float)

    current_oi = df["oi"].iloc[-1]
    oi_4h_ago = df["oi"].iloc[-5] if len(df) >= 5 else df["oi"].iloc[0]
    oi_24h_ago = df["oi"].iloc[-25] if len(df) >= 25 else df["oi"].iloc[0]

    chg_4h = (current_oi / oi_4h_ago - 1) * 100 if oi_4h_ago else 0
    chg_24h = (current_oi / oi_24h_ago - 1) * 100 if oi_24h_ago else 0

    labels = [dt.strftime("%m/%d %H:%M") for dt in df["dt"]]
    oi_vals = [round(float(v) / 1e9, 3) for v in df["oi"]]  # in billions

    return f"""
    <div class="grid grid-3">
      {card("當前 OI", f'{current_oi/1e9:.2f}B', "")}
      {card("4h 變化", f'{chg_4h:+.2f}%', "",
            "#4caf50" if chg_4h > 0 else "#f44336")}
      {card("24h 變化", f'{chg_24h:+.2f}%', "",
            "#4caf50" if chg_24h > 0 else "#f44336")}
    </div>
    <div style="position:relative;height:160px">
      <canvas id="oiChart"></canvas>
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('oiChart').getContext('2d'), {{
        type: 'line',
        data: {{
          labels: {_json.dumps(labels)},
          datasets: [{{ label: 'OI (B)', data: {_json.dumps(oi_vals)},
            borderColor: '#e040fb', backgroundColor: 'rgba(224,64,251,0.05)',
            fill: true, tension: 0.3, borderWidth: 2, pointRadius: 0 }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#e040fb', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }},
                  title: {{ display: true, text: 'OI (Billion)', color: '#e040fb' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Long/Short Ratio ─────────────────────────────────────────────────

def _build_ls_ratio() -> str:
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, cg_ls_ratio, cg_top_trader_ls
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 48 HOUR)
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">{e}</div>'

    if not rows:
        return '<div style="color:#666">無數據</div>'

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"])

    has_ls = "cg_ls_ratio" in df.columns and df["cg_ls_ratio"].notna().any()
    has_top = "cg_top_trader_ls" in df.columns and df["cg_top_trader_ls"].notna().any()

    if not has_ls:
        return '<div style="color:#666">多空比數據不可用</div>'

    df["ls"] = df["cg_ls_ratio"].astype(float)
    current_ls = df["ls"].iloc[-1]

    labels = [dt.strftime("%m/%d %H:%M") for dt in df["dt"]]
    ls_vals = df["ls"].round(3).tolist()

    datasets = f"""{{ label: 'L/S Ratio', data: {_json.dumps(ls_vals)},
        borderColor: '#58a6ff', borderWidth: 2, pointRadius: 0, tension: 0.3 }}"""

    if has_top:
        df["top_ls"] = df["cg_top_trader_ls"].astype(float)
        top_vals = df["top_ls"].round(3).tolist()
        datasets += f""",{{ label: '大戶 L/S', data: {_json.dumps(top_vals)},
            borderColor: '#ff9800', borderWidth: 2, pointRadius: 0, tension: 0.3 }}"""

    ls_sentiment = "偏多" if current_ls > 1.05 else "偏空" if current_ls < 0.95 else "均衡"
    ls_color = "#4caf50" if current_ls > 1.05 else "#f44336" if current_ls < 0.95 else "#8b949e"

    return f"""
    <div class="grid grid-3">
      {card("多空比", f'{current_ls:.3f}', ls_sentiment, ls_color)}
      {card("48h 最高", f'{df["ls"].max():.3f}', "")}
      {card("48h 最低", f'{df["ls"].min():.3f}', "")}
    </div>
    <div style="position:relative;height:160px">
      <canvas id="lsChart"></canvas>
    </div>
    <script>
    (function() {{
      new Chart(document.getElementById('lsChart').getContext('2d'), {{
        type: 'line',
        data: {{ labels: {_json.dumps(labels)}, datasets: [{datasets}] }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{ labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }},
            annotation: {{ annotations: {{
              eq: {{ type: 'line', yMin: 1, yMax: 1,
                     borderColor: '#666', borderWidth: 1, borderDash: [4,4] }}
            }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#8b949e', font: {{ size: 9 }}, maxRotation: 45 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#58a6ff', font: {{ size: 9 }} }}, grid: {{ color: '#21262d' }} }}
          }}
        }}
      }});
    }})();
    </script>"""


# ── Cross-Timeframe Consistency ──────────────────────────────────────

def _build_cross_timeframe() -> str:
    """Compare 1h, 4h, 24h directional tendency."""
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt, close, pred_direction_code, pred_return_4h
                FROM indicator_history
                WHERE dt >= DATE_SUB(NOW(), INTERVAL 25 HOUR)
                ORDER BY dt ASC
            """)
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f'<div style="color:#666">{e}</div>'

    if len(rows) < 24:
        return '<div style="color:#666">數據不足 (需 24h)</div>'

    df = pd.DataFrame(rows)
    df["close"] = df["close"].astype(float)
    df["pred_direction_code"] = df["pred_direction_code"].astype(int)

    # Price-based trend
    close_now = df["close"].iloc[-1]
    close_1h = df["close"].iloc[-2] if len(df) >= 2 else close_now
    close_4h = df["close"].iloc[-5] if len(df) >= 5 else close_now
    close_24h = df["close"].iloc[0]

    ret_1h = (close_now / close_1h - 1) * 100
    ret_4h = (close_now / close_4h - 1) * 100
    ret_24h = (close_now / close_24h - 1) * 100

    # Model directional tendency
    last_1 = df["pred_direction_code"].iloc[-1]
    last_4_avg = df["pred_direction_code"].tail(4).mean()
    last_24_avg = df["pred_direction_code"].mean()

    def _arrow(val):
        if val > 0.3:
            return '<span style="color:#4caf50;font-size:16px">&#9650;</span> UP'
        elif val < -0.3:
            return '<span style="color:#f44336;font-size:16px">&#9660;</span> DOWN'
        return '<span style="color:#8b949e">&#9644;</span> NEUTRAL'

    def _ret_color(val):
        return "#4caf50" if val > 0 else "#f44336"

    # Consistency check
    signs = [np.sign(ret_1h), np.sign(ret_4h), np.sign(ret_24h)]
    model_signs = [np.sign(last_1), np.sign(last_4_avg), np.sign(last_24_avg)]

    price_consistent = len(set(s for s in signs if s != 0)) <= 1
    model_consistent = len(set(s for s in model_signs if s != 0)) <= 1
    price_model_agree = (np.sign(ret_4h) == np.sign(last_4_avg)) or last_4_avg == 0

    consistency_score = sum([price_consistent, model_consistent, price_model_agree])
    cons_label = {3: "高度一致", 2: "部分一致", 1: "分歧", 0: "強烈分歧"}
    cons_color = {3: "#4caf50", 2: "#ff9800", 1: "#f44336", 0: "#f44336"}

    return f"""
    <div style="margin-bottom:8px">
      一致性: <span style="color:{cons_color.get(consistency_score, '#666')};font-weight:600">
        {cons_label.get(consistency_score, '?')}</span>
    </div>
    <table>
      <tr><th>時間框架</th><th>價格變化</th><th>模型方向</th></tr>
      <tr><td>1h</td>
          <td style="color:{_ret_color(ret_1h)}">{ret_1h:+.2f}%</td>
          <td>{_arrow(last_1)}</td></tr>
      <tr><td>4h</td>
          <td style="color:{_ret_color(ret_4h)}">{ret_4h:+.2f}%</td>
          <td>{_arrow(last_4_avg)}</td></tr>
      <tr><td>24h</td>
          <td style="color:{_ret_color(ret_24h)}">{ret_24h:+.2f}%</td>
          <td>{_arrow(last_24_avg)}</td></tr>
    </table>
    <div style="color:#8b949e;font-size:10px;margin-top:6px">
      模型方向 = 該時段內預測方向的平均值
    </div>"""
