"""Build a demo HTML chart of the LDC swing strategy on 5.5 months of data,
without using DB (uses backtest output).  Lets user preview the chart
before production accumulates real signals."""
from __future__ import annotations
import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd
from datetime import timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.dual_model.ldc_swing_backtest import simulate_swing, load_data

OUT_PATH = PROJECT_ROOT / "ldc_swing_demo.html"


def main():
    ldc, klines, _ = load_data()
    trades = simulate_swing(
        ldc, klines,
        exit_mode="dynamic_cross", min_hold_bars=4,
        cost=0.0005,
    )

    # Build candle + marker arrays for last 90 days
    last_90 = klines.tail(90 * 24)
    candle_data = []
    for dt, row in last_90.iterrows():
        utc_ts = int(dt.timestamp())
        ts_disp = utc_ts + 8 * 3600
        candle_data.append({
            "time": ts_disp,
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
        })

    cutoff = last_90.index[0]
    trades_window = trades[trades["entry_ts"] >= cutoff]

    markers = []
    n_long = n_short = n_wins = n_losses = 0
    cum_net = 0.0
    for _, tr in trades_window.iterrows():
        et_disp = int(tr["entry_ts"].timestamp()) + 8 * 3600
        xt_disp = int(tr["exit_ts"].timestamp()) + 8 * 3600
        d = tr["direction"]
        if d == "LONG":
            n_long += 1
            markers.append({"time": et_disp, "position": "belowBar",
                             "color": "#26a69a", "shape": "arrowUp",
                             "text": "L", "size": 2})
        else:
            n_short += 1
            markers.append({"time": et_disp, "position": "aboveBar",
                             "color": "#ef5350", "shape": "arrowDown",
                             "text": "S", "size": 2})
        won = bool(tr["win"])
        if won: n_wins += 1
        else:   n_losses += 1
        ex_color = "#00c853" if won else "#d50000"
        reason = tr["exit_reason"]
        net_disp = float(tr["net_pct"]) * 100
        markers.append({
            "time": xt_disp, "position": "inBar",
            "color": ex_color, "shape": "circle",
            "text": f"{reason} {net_disp:+.2f}%", "size": 1,
        })
        cum_net += net_disp

    markers.sort(key=lambda m: m["time"])
    wr = n_wins / (n_wins + n_losses) * 100 if (n_wins + n_losses) > 0 else 0
    cum_lev = cum_net * 3
    last_dt = klines.index[-1].tz_localize("UTC").astimezone(timezone(timedelta(hours=8))) \
                if klines.index[-1].tz is None \
                else klines.index[-1].astimezone(timezone(timedelta(hours=8)))
    last_time_str = last_dt.strftime("%Y-%m-%d %H:%M UTC+8")
    last_price = float(klines["close"].iloc[-1])

    candle_json = json.dumps(candle_data)
    markers_json = json.dumps(markers)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LDC Swing Demo (90d)</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #b0b8c4; font-family: -apple-system, sans-serif; overflow: hidden; }}
  #header {{
    padding: 8px 12px; font-size: 12px; color: #7a828e;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid #1c222b;
  }}
  #header .title {{ color: #fff; font-weight: 600; font-size: 13px; }}
  #header .info  {{ color: #58a6ff; }}
  #stats {{
    padding: 6px 12px; font-size: 11px; background: #11161e;
    border-bottom: 1px solid #1c222b; display: flex; gap: 16px; flex-wrap: wrap;
  }}
  #stats .pill {{ padding: 2px 8px; border-radius: 10px; background: #1c222b; }}
  #stats .ok  {{ color: #26a69a; }}
  #stats .bad {{ color: #ef5350; }}
  #chart {{ position: absolute; left: 0; right: 0; bottom: 0; top: 60px; }}
  .legend {{
    position: absolute; left: 8px; top: 70px; font-size: 10px;
    color: #b0b8c4; z-index: 10; pointer-events: none;
    background: rgba(15,20,28,0.85); padding: 6px 10px; border-radius: 4px;
  }}
  .legend span {{ margin-right: 10px; }}
</style>
</head>
<body>
<div id="header">
  <div class="title">LDC Swing (jdehorty + min hold 4h, 3x leverage)</div>
  <div class="info">{last_time_str} | ${last_price:,.0f}</div>
</div>
<div id="stats">
  <span class="pill">Trades: {len(trades_window)}</span>
  <span class="pill ok">LONG: {n_long}</span>
  <span class="pill bad">SHORT: {n_short}</span>
  <span class="pill">WR: {wr:.1f}%</span>
  <span class="pill {'ok' if cum_net > 0 else 'bad'}">Cum (1x): {cum_net:+.2f}%</span>
  <span class="pill {'ok' if cum_lev > 0 else 'bad'}">Cum (3x): {cum_lev:+.2f}%</span>
  <span class="pill">Window: 90d demo</span>
</div>
<div class="legend">
  <span style="color:#26a69a;">▲ LONG entry</span>
  <span style="color:#ef5350;">▼ SHORT entry</span>
  <span style="color:#00c853;">● Win exit</span>
  <span style="color:#d50000;">● Loss exit</span>
</div>
<div id="chart"></div>

<script>
const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  layout: {{ background: {{ color: '#0d1117' }}, textColor: '#b0b8c4' }},
  grid: {{ vertLines: {{ color: '#1c222b' }}, horzLines: {{ color: '#1c222b' }} }},
  rightPriceScale: {{ borderColor: '#1c222b' }},
  timeScale: {{ borderColor: '#1c222b', timeVisible: true, secondsVisible: false }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
}});
const candleSeries = chart.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350',
  borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});
candleSeries.setData({candle_json});
candleSeries.setMarkers({markers_json});
chart.timeScale().fitContent();
window.addEventListener('resize', () => chart.applyOptions({{
  width: window.innerWidth, height: window.innerHeight - 60,
}}));
</script>
</body>
</html>"""
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"Trades in 90d window: {len(trades_window)}  WR {wr:.1f}%  "
          f"Cum {cum_net:+.2f}% (1x) / {cum_lev:+.2f}% (3x)")


if __name__ == "__main__":
    main()
