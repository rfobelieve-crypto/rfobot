"""
Interactive indicator chart — TradingView Lightweight Charts.

Mobile-optimized: pinch-to-zoom, drag-to-scroll, crosshair on touch.
Same 4-panel layout as Telegram static chart.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def render_interactive_chart(ind: pd.DataFrame, last_n: int = 200) -> str:
    """Render interactive chart using TradingView Lightweight Charts. Returns HTML."""
    sig = ind.tail(last_n).copy()
    sig = sig.dropna(subset=["open", "high", "low", "close"])

    if len(sig) == 0:
        return "<h3>No data</h3>"

    # Ensure UTC index, then offset for UTC+8 display
    from datetime import timezone, timedelta
    TZ_UTC8 = timezone(timedelta(hours=8))
    try:
        if sig.index.tz is None:
            sig.index = sig.index.tz_localize("UTC")
    except Exception:
        pass

    has_mag = "mag_pred" in sig.columns and sig["mag_pred"].notna().any()
    has_regime = "regime" in sig.columns and sig["regime"].notna().any()

    REGIME_COLORS = {
        "TRENDING_BULL": "#26a69a",
        "TRENDING_BEAR": "#ef5350",
        "CHOPPY":        "#78828f",
        "WARMUP":        "#3a3f4a",
    }

    # ── Build data arrays ──
    candle_data = []
    conf_data = []
    mag_data = []
    regime_data = []
    markers = []

    for i, (dt, row) in enumerate(sig.iterrows()):
        # TradingView treats timestamps as UTC for display.
        # Add 8h offset so the displayed time reads as UTC+8.
        utc_ts = int(dt.timestamp()) if dt.tzinfo else int(dt.replace(tzinfo=timezone.utc).timestamp())
        ts = utc_ts + 8 * 3600
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

        candle_data.append({"time": ts, "open": o, "high": h, "low": l, "close": c})

        # Confidence → color
        conf_val = float(row.get("confidence_score", 0) or 0)
        if conf_val >= 70:
            conf_color = "#e040fb"
        elif conf_val >= 40:
            conf_color = "#6a1b9a"
        else:
            conf_color = "#1a1a2e"
        conf_data.append({"time": ts, "value": conf_val / 100, "color": conf_color})

        # Magnitude — colour by regression lean sign + tier shade.
        # Prefer dir_pred_ret so every bar gets a directional colour; fall
        # back to pred_direction for legacy binary compat.
        if has_mag:
            # Always positive; direction is encoded purely by colour.
            mag_val = abs(float(row.get("mag_pred", 0) or 0)) * 100
            dpr = row.get("dir_pred_ret")
            if dpr is not None and pd.notna(dpr):
                sgn = 1 if float(dpr) > 0 else (-1 if float(dpr) < 0 else 0)
            else:
                d = str(row.get("pred_direction", "NEUTRAL"))
                sgn = 1 if d == "UP" else (-1 if d == "DOWN" else 0)
            s = str(row.get("strength_score", "Weak"))
            if s == "Strong":
                mag_color = "#004d40" if sgn > 0 else ("#b71c1c" if sgn < 0 else "#bdbdbd")
            elif s == "Moderate":
                mag_color = "#66bb6a" if sgn > 0 else ("#ef5350" if sgn < 0 else "#bdbdbd")
            else:
                mag_color = "#bdbdbd"
            mag_data.append({"time": ts, "value": round(mag_val, 4), "color": mag_color})

        # Regime strip (value=1 for all bars, color encodes regime)
        if has_regime:
            reg_val = str(row.get("regime", "CHOPPY"))
            regime_data.append({
                "time": ts,
                "value": 1,
                "color": REGIME_COLORS.get(reg_val, REGIME_COLORS["CHOPPY"]),
            })

        # Direction markers (Strong + Moderate)
        strength = str(row.get("strength_score", ""))
        direction = str(row.get("pred_direction", ""))
        if strength in ("Strong", "Moderate") and direction in ("UP", "DOWN"):
            is_strong = (strength == "Strong")
            if direction == "UP":
                markers.append({
                    "time": ts,
                    "position": "belowBar",
                    "color": "#004d40" if is_strong else "#66bb6a",
                    "shape": "arrowUp",
                    "text": "",
                    "size": 2 if is_strong else 1,
                })
            else:
                markers.append({
                    "time": ts,
                    "position": "aboveBar",
                    "color": "#b71c1c" if is_strong else "#ef5350",
                    "shape": "arrowDown",
                    "text": "",
                    "size": 2 if is_strong else 1,
                })

    last_dt = sig.index[-1]
    if last_dt.tzinfo:
        last_dt = last_dt.astimezone(TZ_UTC8)
    last_time = last_dt.strftime("%Y-%m-%d %H:%M UTC+8")
    last_price = float(sig["close"].iloc[-1])
    last_dir = str(sig.get("pred_direction", pd.Series("?")).iloc[-1])
    last_conf = float(sig.get("confidence_score", pd.Series(0)).iloc[-1] or 0)
    last_str = str(sig.get("strength_score", pd.Series("?")).iloc[-1])
    last_reg = str(sig.get("regime", pd.Series("?")).iloc[-1])
    # Short display: TRENDING_BULL → BULL, etc.
    reg_short = last_reg.replace("TRENDING_", "")
    reg_color = {
        "TRENDING_BULL": "#26a69a",
        "TRENDING_BEAR": "#ef5350",
        "CHOPPY":        "#78828f",
        "WARMUP":        "#999999",
    }.get(last_reg, "#999999")

    # Height ratios for panels. Reserve 5% for regime strip if present.
    # BBP panel was removed 2026-04-22 — its 15% height returned to price.
    regime_pct = 5 if has_regime else 0
    if has_mag:
        price_pct = 75 - regime_pct
        mag_pct_h = 15
        conf_pct = 10
    else:
        price_pct = 90 - regime_pct
        mag_pct_h = 0
        conf_pct = 10

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>BTC Indicator</title>
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
  #header .info {{ color: #58a6ff; }}
  .chart-label {{
    position: absolute; left: 4px; top: 2px; font-size: 10px;
    color: #7a828e; z-index: 10; pointer-events: none;
  }}
  .chart-wrapper {{ position: relative; }}
  #footer {{
    padding: 4px 12px; font-size: 10px; color: #7a828e;
    text-align: right; border-top: 1px solid #1c222b;
  }}
</style>
</head>
<body>
<div id="header">
  <span class="title">BTC Market Intelligence Indicator (4h prediction)</span>
  <span class="info">{last_dir} | Conf {last_conf:.0f}% ({last_str}) | <span style="color: {reg_color}; font-weight: 600;">{reg_short}</span> | ${last_price:,.0f}</span>
  <span>{last_time}</span>
</div>
<div class="chart-wrapper" id="conf-wrapper">
  <div class="chart-label">Confidence</div>
  <div id="chart-conf"></div>
</div>
{"<div class='chart-wrapper' id='regime-wrapper'><div class='chart-label'>Regime</div><div id='chart-regime'></div></div>" if has_regime else ""}
<div class="chart-wrapper" id="price-wrapper">
  <div class="chart-label">Price (USD)</div>
  <div id="chart-price"></div>
</div>
{"<div class='chart-wrapper' id='mag-wrapper'><div class='chart-label'>Magnitude (%)</div><div id='chart-mag'></div></div>" if has_mag else ""}
<div id="footer">source@rfo</div>

<script>
const candleData = {json.dumps(candle_data)};
const confData = {json.dumps(conf_data)};
const magData = {json.dumps(mag_data)};
const regimeData = {json.dumps(regime_data)};
const markers = {json.dumps(markers)};
const hasMag = {'true' if has_mag else 'false'};
const hasRegime = {'true' if has_regime else 'false'};

const BG = '#0d1117';
const CARD = '#161b22';
const GRID = '#1c222b';
const TEXT = '#7a828e';

const headerH = 36;
const footerH = 24;
const totalH = window.innerHeight - headerH - footerH;
// see AXIS_W comment below — declared here so makeChart default can use it
const AXIS_W_DEFAULT = 60;
const confH = Math.round(totalH * 0.{conf_pct:02d});
const regimeH = hasRegime ? Math.round(totalH * 0.{regime_pct:02d}) : 0;
const priceH = Math.round(totalH * 0.{price_pct:02d});
const magH = hasMag ? Math.round(totalH * 0.{mag_pct_h:02d}) : 0;

function makeChart(container, height, opts) {{
  const el = document.getElementById(container);
  el.style.height = height + 'px';
  const chart = LightweightCharts.createChart(el, {{
    width: window.innerWidth,
    height: height,
    layout: {{ background: {{ color: BG }}, textColor: TEXT, fontSize: 10 }},
    grid: {{ vertLines: {{ color: GRID }}, horzLines: {{ color: GRID }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{
      timeVisible: true, secondsVisible: false,
      borderColor: GRID,
    }},
    rightPriceScale: {{ borderColor: GRID, minimumWidth: AXIS_W_DEFAULT }},
    ...opts,
  }});
  return chart;
}}

// Align every chart's price-axis width to AXIS_W so plot areas line up
// across panes. Without this, each pane's axis width depends on its own
// tick label length (e.g. price "78119.57" ≫ mag "0.500"), producing
// visual drift between the Confidence heatmap, regime strip,
// candlesticks, and magnitude panels. minimumWidth is a Lightweight
// Charts v4.1+ hint; ignored gracefully on older versions.
const AXIS_W = 60;

// ── Confidence chart ──
const confChart = makeChart('chart-conf', confH, {{
  rightPriceScale: {{ visible: true, borderVisible: false, drawTicks: false,
    scaleMargins: {{ top: 0, bottom: 0 }}, minimumWidth: AXIS_W }},
}});
const confSeries = confChart.addHistogramSeries({{
  priceFormat: {{ type: 'custom', formatter: (v) => (v * 100).toFixed(0) + '%' }},
  priceLineVisible: false,
  lastValueVisible: false,
}});
confSeries.setData(confData);

// ── Regime strip ──
// Histogram series with per-bar color encoding the regime. Value=1 for
// every bar, so the visual is a uniform color band where the color tells
// the regime. Price axis kept VISIBLE (but empty label) to reserve the
// same horizontal width as sibling charts — otherwise the strip's
// plot area is ~50px wider and the bars drift out of alignment with
// the Confidence heatmap and candlestick above/below.
let regimeChart = null;
let regimeSeries = null;
if (hasRegime) {{
  regimeChart = makeChart('chart-regime', regimeH, {{
    rightPriceScale: {{
      visible: true, borderVisible: false, drawTicks: false,
      ticksVisible: false, entireTextOnly: true,
      scaleMargins: {{ top: 0, bottom: 0 }}, minimumWidth: AXIS_W,
    }},
    timeScale: {{ visible: false, borderColor: GRID }},
  }});
  regimeSeries = regimeChart.addHistogramSeries({{
    priceFormat: {{ type: 'custom', formatter: () => '' }},
    priceLineVisible: false,
    lastValueVisible: false,
    base: 0,
  }});
  regimeSeries.setData(regimeData);
}}

// ── Price chart ──
const priceChart = makeChart('chart-price', priceH, {{
  rightPriceScale: {{ autoScale: true, minimumWidth: AXIS_W }},
}});
const candleSeries = priceChart.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350',
  borderUpColor: '#26a69a', borderDownColor: '#ef5350',
  wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});
candleSeries.setData(candleData);
candleSeries.setMarkers(markers);

// ── Magnitude chart ──
let magChart = null;
let magSeries = null;
if (hasMag) {{
  magChart = makeChart('chart-mag', magH);
  magSeries = magChart.addHistogramSeries({{
    // Finer resolution: 3 decimals + 0.001 minMove so small mag bars
    // and their y-axis ticks stay readable when typical |mag| < 0.5%.
    priceFormat: {{ type: 'price', precision: 3, minMove: 0.001 }},
    priceLineVisible: false,
    lastValueVisible: false,
  }});
  magSeries.setData(magData);

  // Reference lines: p80 / p90 of mag within the window.
  const absMag = magData.map(d => d.value).filter(v => v > 1e-9);
  if (absMag.length >= 10) {{
    absMag.sort((a, b) => a - b);
    const q = (arr, p) => arr[Math.floor(p * (arr.length - 1))];
    const p80 = q(absMag, 0.80);
    const p90 = q(absMag, 0.90);
    [
      [p80, '#f9a825', 'p80'],
      [p90, '#ef6c00', 'p90'],
    ].forEach(([price, color, title]) => {{
      magSeries.createPriceLine({{
        price: price,
        color: color,
        lineWidth: 1,
        lineStyle: 2,  // dashed
        axisLabelVisible: true,
        title: title,
      }});
    }});
  }}
}}

// ── Sync all charts (scroll + zoom) ──
const charts = [confChart, regimeChart, priceChart, magChart].filter(Boolean);

let isSyncing = false;
charts.forEach((chart, idx) => {{
  chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
    if (isSyncing) return;
    isSyncing = true;
    charts.forEach((other, j) => {{
      if (j !== idx && range) {{
        other.timeScale().setVisibleLogicalRange(range);
      }}
    }});
    isSyncing = false;
  }});
}});

// ── Sync crosshair across all panels ──
const chartSeriesPairs = [
  [confChart, confSeries, 0.5],
  hasRegime ? [regimeChart, regimeSeries, 1] : null,
  [priceChart, candleSeries, null],
  hasMag ? [magChart, magSeries, 0] : null,
].filter(Boolean);

let isSyncingCH = false;
chartSeriesPairs.forEach(([srcChart], idx) => {{
  srcChart.subscribeCrosshairMove((param) => {{
    if (isSyncingCH) return;
    isSyncingCH = true;
    chartSeriesPairs.forEach(([dstChart, dstSeries, defaultPrice], j) => {{
      if (j !== idx) {{
        if (param.time) {{
          try {{
            // For candlestick, use close price; for histograms use default
            let price = defaultPrice !== null ? defaultPrice : 0;
            if (dstSeries === candleSeries && param.seriesData) {{
              const srcData = param.seriesData.get(chartSeriesPairs[idx][1]);
              if (srcData && srcData.close) price = srcData.close;
            }}
            dstChart.setCrosshairPosition(price, param.time, dstSeries);
          }} catch(e) {{}}
        }} else {{
          try {{ dstChart.clearCrosshairPosition(); }} catch(e) {{}}
        }}
      }}
    }});
    isSyncingCH = false;
  }});
}});

// ── Resize ──
window.addEventListener('resize', () => {{
  const totalH2 = window.innerHeight - headerH - footerH;
  const newConfH = Math.round(totalH2 * 0.{conf_pct:02d});
  const newPriceH = Math.round(totalH2 * 0.{price_pct:02d});
  const newMagH = hasMag ? Math.round(totalH2 * 0.{mag_pct_h:02d}) : 0;
  confChart.resize(window.innerWidth, newConfH);
  priceChart.resize(window.innerWidth, newPriceH);
  if (magChart) magChart.resize(window.innerWidth, newMagH);
}});
</script>
</body>
</html>"""
    return html
