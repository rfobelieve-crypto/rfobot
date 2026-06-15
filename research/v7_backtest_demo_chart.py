"""
Generate v7_backtest_demo.html — interactive chart of the V7 backtest trades.

The live indicator chart only shows the last ~200 bars, so the V7 BACKTEST
trades (2025-12 → 2026-04, WF-OOS) cannot appear on it. This produces a
standalone demo chart covering the full backtest window with every V7
entry/exit marked — the same idea as the existing ldc_swing_demo.html.

Source: research/results/v7_3atr_1x_trades.csv  (from v71_v7_sizing_1x.py)
Output: v7_backtest_demo.html  (repo root)
"""
from __future__ import annotations
import json
from datetime import timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "research" / "results" / "v7_3atr_1x_trades.csv"
KLINES = ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
OUT = ROOT / "v7_backtest_demo.html"

REASON_SHORT = {"trail_stop": "trail", "time_cap": "cap", "opp_signal": "sig"}


def main():
    tr = pd.read_csv(TRADES, parse_dates=["entry_ts", "exit_ts"])
    k = pd.read_parquet(KLINES)[["open", "high", "low", "close"]].dropna()
    if k.index.tz is not None:
        k.index = k.index.tz_convert("UTC").tz_localize(None)
    k = k[~k.index.duplicated(keep="last")].sort_index()

    # window: backtest span + padding
    lo = tr["entry_ts"].min() - pd.Timedelta(days=3)
    hi = tr["exit_ts"].max() + pd.Timedelta(days=3)
    kw = k[(k.index >= lo) & (k.index <= hi)]

    candles, valid_ts = [], set()
    for dt, row in kw.iterrows():
        ts = int(dt.replace(tzinfo=timezone.utc).timestamp()) + 8 * 3600
        valid_ts.add(ts)
        candles.append({"time": ts, "open": float(row["open"]),
                        "high": float(row["high"]), "low": float(row["low"]),
                        "close": float(row["close"])})

    def _ts(dt):
        return int(pd.Timestamp(dt).replace(tzinfo=timezone.utc).timestamp()) \
            + 8 * 3600

    markers = []
    for _, t in tr.iterrows():
        up = t["direction"] == "UP"
        e_ts = _ts(t["entry_ts"])
        if e_ts in valid_ts:
            markers.append({
                "time": e_ts,
                "position": "belowBar" if up else "aboveBar",
                "color": "#29b6f6", "shape": "circle",
                "text": "IN", "size": 1,
            })
        x_ts = _ts(t["exit_ts"])
        if x_ts in valid_ts:
            won = float(t["equity_ret_pct"]) > 0
            markers.append({
                "time": x_ts,
                "position": "aboveBar" if up else "belowBar",
                "color": "#26a69a" if won else "#ef5350",
                "shape": "square",
                "text": REASON_SHORT.get(t["exit_reason"], t["exit_reason"]),
                "size": 1,
            })
    markers.sort(key=lambda m: m["time"])

    n = len(tr)
    wr = (tr["equity_ret_pct"] > 0).mean() * 100
    roi = (tr["equity_after"].iloc[-1] / 1000.0 - 1.0) * 100
    summary = (f"V7 Backtest Demo — {n} trades | WR {wr:.1f}% | "
               f"ROI {roi:+.1f}% (1000u, 2% risk, 1x) | "
               f"{tr['entry_ts'].min():%Y-%m-%d} → {tr['exit_ts'].max():%Y-%m-%d} "
               f"| WF-OOS backtest")

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V7 Backtest Demo</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d1117; color:#b0b8c4; font-family:-apple-system,sans-serif; overflow:hidden; }}
  #header {{ padding:8px 12px; font-size:12px; border-bottom:1px solid #1c222b; }}
  #header b {{ color:#fff; }}
  #legend {{ padding:4px 12px; font-size:10px; color:#7a828e; }}
  #footer {{ padding:4px 12px; font-size:10px; color:#7a828e; text-align:right; }}
</style></head>
<body>
<div id="header"><b>{summary}</b></div>
<div id="legend">🔵 IN = V7 進場 &nbsp;|&nbsp; 🟩 綠 = 獲利出場 &nbsp;|&nbsp;
  🟥 紅 = 虧損出場 &nbsp;|&nbsp; 出場標籤: trail=移動停損 / cap=72h / sig=反向訊號</div>
<div id="chart"></div>
<div id="footer">source@rfo — research/v7_backtest_demo_chart.py</div>
<script>
const candleData = {json.dumps(candles)};
const markers = {json.dumps(markers)};
const el = document.getElementById('chart');
const H = window.innerHeight - 80;
el.style.height = H + 'px';
const chart = LightweightCharts.createChart(el, {{
  width: window.innerWidth, height: H,
  layout: {{ background: {{ color:'#0d1117' }}, textColor:'#7a828e', fontSize:10 }},
  grid: {{ vertLines:{{ color:'#1c222b' }}, horzLines:{{ color:'#1c222b' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  timeScale: {{ timeVisible:true, secondsVisible:false, borderColor:'#1c222b' }},
}});
const s = chart.addCandlestickSeries({{
  upColor:'#26a69a', downColor:'#ef5350',
  borderUpColor:'#26a69a', borderDownColor:'#ef5350',
  wickUpColor:'#26a69a', wickDownColor:'#ef5350',
}});
s.setData(candleData);
s.setMarkers(markers);
chart.timeScale().fitContent();
window.addEventListener('resize', () => {{
  const h = window.innerHeight - 80;
  el.style.height = h + 'px';
  chart.resize(window.innerWidth, h);
}});
</script>
</body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(candles)} candles, {len(markers)} markers, "
          f"{n} trades)")


if __name__ == "__main__":
    main()
