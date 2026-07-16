"""Cancel-flow interactive review chart — research/eyeball tool (NOT production).

Same data as plot_cancel_flow.py but rendered as a TradingView Lightweight
Charts HTML (the /ichart pattern): 1m candlesticks + cancel skew + cancel
intensity in three synced panes, so撤單 episodes can be replayed against
real K-bars at any zoom level (覆盤 use).

Price bars come from Binance 1m klines (public REST) — orderbook_snapshots_1m
only stores one mid per minute, no OHLC. v7 Strong signals are overlaid as
markers (read-only, best-effort), mirroring the static monitor.

DISCIPLINE: monitoring/intuition tool, not a signal. Edge unproven until the
2026-08-10 cancel_lead_ic verdict. Read-only; no production imports; does not
touch the production static/interactive charts.

Usage:
    python research/cancel_flow_interactive.py              # full depth era
    python research/cancel_flow_interactive.py --hours 48   # last 48h only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from shared.db import get_db_conn

SMOOTH_MIN = 15
TZ_OFFSET_S = 8 * 3600  # display as UTC+8, same convention as chart_interactive
OUT = PROJECT_ROOT / "research" / "results" / "cancel_flow_review.html"


def _q(conn, sql: str, params=None) -> pd.DataFrame:
    """DB → DataFrame without pd.read_sql: its handling of DictCursor rows
    differs across pandas versions (the container's newer pandas turned the
    column ALIAS into row values). dict rows via pd.DataFrame() are stable."""
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        rows = cur.fetchall() or []
    return pd.DataFrame(rows)


def load_depth(hours: int | None) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        dd = _q(conn,
            "SELECT minute_start_ms ms, bid_add_qty, bid_cancel_qty, "
            "ask_add_qty, ask_cancel_qty FROM depth_deltas_1m "
            "WHERE canonical_symbol='BTC-USD' AND exchange='binance' "
            "ORDER BY minute_start_ms")
    finally:
        conn.close()
    if dd.empty:
        return dd
    for c in ("bid_add_qty", "bid_cancel_qty", "ask_add_qty", "ask_cancel_qty"):
        dd[c] = dd[c].astype(float)
    # Newer pandas (Railway image) reads DB numerics as arrow-backed str;
    # local Windows pandas reads them as ints. Coerce so int64 works on both.
    dd["ms"] = pd.to_numeric(dd["ms"]).astype("int64")
    if hours:
        dd = dd[dd["ms"] >= dd["ms"].max() - hours * 3600_000]
    return dd.reset_index(drop=True)


def fetch_klines_1m(start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cur = start_ms
    while cur < end_ms:
        resp = requests.get("https://api.binance.com/api/v3/klines", params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": cur, "endTime": end_ms, "limit": 1000,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 60_000
        time.sleep(0.15)
    df = pd.DataFrame(rows, columns=[
        "open_time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbb", "tbq", "ig"])
    for col in ("o", "h", "l", "c"):
        df[col] = df[col].astype(float)
    return df[["open_time", "o", "h", "l", "c"]]


def load_strong_signals(start_ms: int, end_ms: int) -> list[dict]:
    try:
        conn = get_db_conn()
        try:
            sig = _q(conn,
                "SELECT signal_time, direction FROM tracked_signals "
                "WHERE strength='Strong' AND direction IN ('UP','DOWN') "
                "AND signal_time >= %s AND signal_time <= %s",
                params=(str(pd.Timestamp(start_ms, unit="ms")),
                        str(pd.Timestamp(end_ms, unit="ms"))))
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"(signal overlay skipped: {e})")
        return []
    out = []
    for _, r in sig.iterrows():
        ts = int(pd.Timestamp(r["signal_time"]).timestamp()) + TZ_OFFSET_S
        up = r["direction"] == "UP"
        out.append({"time": ts, "position": "belowBar" if up else "aboveBar",
                    "shape": "arrowUp" if up else "arrowDown",
                    "color": "#26a269" if up else "#e01b24",
                    "text": "S"})
    return sorted(out, key=lambda m: m["time"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=0, help="0 = full depth era")
    ap.add_argument("--shock-dots", action="store_true",
                    help="overlay amber dots on candles where cancel volume "
                         ">= 3x trailing-60m median (off by default)")
    args = ap.parse_args()

    dd = load_depth(args.hours or None)
    if dd.empty or len(dd) < 30:
        print(f"depth_deltas too young ({len(dd)} min)")
        return 0

    tot = dd["bid_cancel_qty"] + dd["ask_cancel_qty"]
    skew = (dd["ask_cancel_qty"] - dd["bid_cancel_qty"]) / tot.replace(0, np.nan)
    skew = skew - skew.mean()  # remove structural ask-heavy bias, cf. plot_cancel_flow
    skew_s = skew.rolling(SMOOTH_MIN, min_periods=max(3, SMOOTH_MIN // 3)).mean()
    intensity = tot.rolling(SMOOTH_MIN, min_periods=3).mean()
    # change-magnitude (shock): raw per-minute total vs trailing 60m median —
    # frozen definition shared with cancel_shock_ic.py (registered 2026-07-15)
    shock_base = tot.rolling(60, min_periods=30).median()
    shock = pd.Series(np.where(shock_base > 0, tot / shock_base, np.nan))

    start_ms, end_ms = int(dd["ms"].min()), int(dd["ms"].max()) + 60_000
    print(f"fetching Binance 1m klines {pd.Timestamp(start_ms, unit='ms')} -> "
          f"{pd.Timestamp(end_ms, unit='ms')} UTC ...")
    kl = fetch_klines_1m(start_ms, end_ms)
    print(f"{len(kl)} candles, {len(dd)} depth minutes")

    candles = [{"time": int(t) // 1000 + TZ_OFFSET_S, "open": o, "high": h,
                "low": lo, "close": c}
               for t, o, h, lo, c in kl.itertuples(index=False)]

    skew_bars, int_bars, shock_bars = [], [], []
    for i in range(len(dd)):
        ts = int(dd["ms"].iloc[i]) // 1000 + TZ_OFFSET_S
        z = skew_s.iloc[i]
        if not np.isnan(z):
            deep = abs(z) >= 0.30
            color = (("#26a269" if deep else "#1e5c3f") if z >= 0
                     else ("#e01b24" if deep else "#7a2028"))
            skew_bars.append({"time": ts, "value": round(float(z), 4), "color": color})
        v = intensity.iloc[i]
        if not np.isnan(v):
            int_bars.append({"time": ts, "value": round(float(v), 2),
                             "color": "#4a6b9a"})
        # shock spikes become amber dot markers on the price pane (binary
        # categorical >=3x, no sub-threshold class — keeps the read simple)
        s = shock.iloc[i]
        if not np.isnan(s) and s >= 3.0:
            shock_bars.append({"time": ts, "position": "aboveBar",
                               "shape": "circle", "color": "#f2b544",
                               "text": "", "size": 1})

    markers = load_strong_signals(start_ms, end_ms)
    if args.shock_dots:
        markers = sorted(markers + shock_bars, key=lambda m: m["time"])
        print(f"{len(shock_bars)} shock(>=3x) markers overlaid")
    span_h = (end_ms - start_ms) / 3600_000

    html = HTML_TEMPLATE.format(
        title=f"撤單流覆盤 BTC-USD ({span_h:.0f}h)",
        candles=json.dumps(candles), skew=json.dumps(skew_bars),
        intensity=json.dumps(int_bars), markers=json.dumps(markers),
        span_h=f"{span_h:.0f}", n=len(dd), smooth=SMOOTH_MIN)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"saved -> {OUT}")
    return 0


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  body {{ margin:0; background:#0e1116; color:#e3e3e3;
         font-family:'Microsoft JhengHei',sans-serif; }}
  #hdr {{ padding:8px 14px; font-size:13px; color:#9aa0a6; }}
  #hdr b {{ color:#e3e3e3; }}
  .pane {{ position:relative; }}
  .lbl {{ position:absolute; top:6px; left:10px; z-index:5;
          font-size:12px; color:#9aa0a6; pointer-events:none; }}
</style></head><body>
<div id="hdr"><b>撤單流覆盤 BTC-USD</b> · {span_h}h · n={n} 分鐘 · 撤單面板 {smooth}m 平滑/去均值
 · ▲▼=v7 Strong · 深色柱=|不對稱|≥0.30 · 研究工具非信號 (edge 待 8/10)</div>
<div class="pane"><div class="lbl">價格 1m K棒</div><div id="c1"></div></div>
<div class="pane"><div class="lbl">撤單不對稱 (＋賣側撤多 / －買側撤多)</div><div id="c2"></div></div>
<div class="pane"><div class="lbl">撤單強度 (兩側總量)</div><div id="c3"></div></div>
<script>
const OPTS = {{
  layout: {{ background: {{ color: '#0e1116' }}, textColor: '#9aa0a6' }},
  grid: {{ vertLines: {{ color: '#2a2f38' }}, horzLines: {{ color: '#2a2f38' }} }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  timeScale: {{ timeVisible: true, secondsVisible: false, minBarSpacing: 0.02 }},
}};
function mk(id, h) {{
  const el = document.getElementById(id);
  el.style.height = h + 'px';
  return LightweightCharts.createChart(el, Object.assign({{}}, OPTS,
      {{ width: el.clientWidth || window.innerWidth, height: h }}));
}}
const c1 = mk('c1', Math.max(260, window.innerHeight * 0.42));
const c2 = mk('c2', Math.max(150, window.innerHeight * 0.24));
const c3 = mk('c3', Math.max(110, window.innerHeight * 0.18));

const CANDLES = {candles};
const SKEW = {skew};
const INTEN = {intensity};

const candle = c1.addCandlestickSeries({{
  upColor:'#26a269', downColor:'#e01b24',
  wickUpColor:'#26a269', wickDownColor:'#e01b24', borderVisible:false }});
candle.setData(CANDLES);
candle.setMarkers({markers});

const skew = c2.addHistogramSeries({{ priceFormat: {{ type:'price', precision:2, minMove:0.01 }} }});
skew.setData(SKEW);
[0.30, -0.30].forEach(v => skew.createPriceLine(
  {{ price: v, color:'#555c66', lineWidth:1, lineStyle:2, title:'' }}));
skew.createPriceLine({{ price: 0, color:'#9aa0a6', lineWidth:1, lineStyle:0, title:'' }});

const inten = c3.addHistogramSeries({{ priceFormat: {{ type:'volume' }} }});
inten.setData(INTEN);

// ── pane sync (visible range + crosshair), guarded against feedback ──
const byTime = arr => {{ const m = new Map();
  arr.forEach(d => m.set(d.time, d.value !== undefined ? d.value : d.close));
  return m; }};
const panes = [
  {{ chart: c1, series: candle, map: byTime(CANDLES) }},
  {{ chart: c2, series: skew,   map: byTime(SKEW) }},
  {{ chart: c3, series: inten,  map: byTime(INTEN) }},
];
const charts = panes.map(p => p.chart);
let syncing = false;
charts.forEach(src => {{
  src.timeScale().subscribeVisibleLogicalRangeChange(range => {{
    if (syncing || !range) return;
    syncing = true;
    charts.forEach(dst => {{ if (dst !== src) dst.timeScale().setVisibleLogicalRange(range); }});
    syncing = false;
  }});
}});
let xhairSyncing = false;
panes.forEach(src => {{
  src.chart.subscribeCrosshairMove(param => {{
    if (xhairSyncing) return;
    xhairSyncing = true;
    panes.forEach(dst => {{
      if (dst.chart === src.chart) return;
      if (param.time !== undefined) {{
        const v = dst.map.get(param.time);
        dst.chart.setCrosshairPosition(v !== undefined ? v : 0, param.time, dst.series);
      }} else {{
        dst.chart.clearCrosshairPosition();
      }}
    }});
    xhairSyncing = false;
  }});
}});
window.addEventListener('resize', () => charts.forEach(c =>
    c.applyOptions({{ width: window.innerWidth }})));
c1.timeScale().fitContent();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
