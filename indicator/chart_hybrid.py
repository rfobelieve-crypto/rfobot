"""
Hybrid v9+LDC signal chart — standalone interactive (TradingView Lightweight).

Independent from indicator/chart_interactive.py — different endpoint, same
candle style.  Marks:

  LONG entry  -> green up-arrow below bar
  SHORT entry -> red  down-arrow above bar
  TP exit     -> green circle
  SL exit     -> red circle
  Timeout exit-> gray circle
  Paused      -> hollow / dimmed arrow (signal recorded but not "live")

Pulls from MySQL hybrid_signals table.  Falls back to "no signals yet"
HTML if table is empty / doesn't exist.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fetch_hybrid_for_chart(since_days: int = 60) -> pd.DataFrame:
    """All ldc_swing_positions (open + closed) in the last N days.

    Returns DataFrame normalized to the chart's expected column names
    (signal_time, direction, entry_price, exit_price, exit_reason,
    bars_held, gross_pct, net_pct_maker, win, paused_at_signal, tp_dist,
    sl_dist, horizon_h).  tp_dist/sl_dist set to 0 since LDC swing has no
    barriers — chart treats 0 as "no line".
    """
    from shared.db import get_db_conn
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    sql = """
        SELECT entry_time AS signal_time, direction,
               entry_price, exit_price, exit_reason, bars_held,
               gross_pct, net_pct_maker, win, paused_at_signal, model_version
        FROM ldc_swing_positions
        WHERE entry_time >= %s
        ORDER BY entry_time ASC
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, (cutoff,))
                rows = cur.fetchall()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return pd.DataFrame()
                raise
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    for c in ("entry_price", "exit_price", "gross_pct", "net_pct_maker"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["win"] = df["win"].astype("Int64")
    df["paused_at_signal"] = df["paused_at_signal"].astype(int)
    df["bars_held"] = df["bars_held"].astype("Int64")
    # LDC swing has no TP/SL barriers — set to 0 so chart skips price lines.
    df["tp_dist"] = 0.0
    df["sl_dist"] = 0.0
    df["horizon_h"] = 0
    # Placeholders for legacy fields chart reads
    df["p_long_win"] = np.nan
    df["p_short_win"] = np.nan
    df["ldc_signal"] = 0
    return df


def _fetch_klines_for_chart(since_days: int = 60) -> pd.DataFrame:
    """Latest klines from parquet for chart backdrop."""
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    p = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
    if not p.exists():
        return pd.DataFrame()
    k = pd.read_parquet(p)[["open", "high", "low", "close"]].dropna()
    if k.index.tz is None:
        k.index = k.index.tz_localize("UTC")
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    return k[k.index >= cutoff]


def _build_demo_signals_from_walkforward() -> pd.DataFrame:
    """Build hybrid_signals-compatible DataFrame from walk-forward OOS parquet
    + LDC signals.  Used for /hybrid-chart?demo=1 so user can preview chart
    appearance before production accumulates real signals."""
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    v9_oos = PROJECT_ROOT / "research" / "results" / "dual_model" / "direction_v9_winrate_H8_oos.parquet"
    ldc_full = PROJECT_ROOT / "research" / "results" / "ldc_signals_full.parquet"
    klines_path = PROJECT_ROOT / "market_data" / "raw_data" / "binance_klines_1h.parquet"
    if not (v9_oos.exists() and ldc_full.exists() and klines_path.exists()):
        return pd.DataFrame()

    v9 = pd.read_parquet(v9_oos)
    v9["ts"] = pd.to_datetime(v9["ts"])
    if v9["ts"].dt.tz is not None:
        v9["ts"] = v9["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    v9 = v9.set_index("ts").sort_index()

    ldc = pd.read_parquet(ldc_full)
    if ldc.index.tz is not None:
        ldc.index = ldc.index.tz_convert("UTC").tz_localize(None)
    v9["ldc_signal"] = ldc["signal"].reindex(v9.index).fillna(0).astype(int)

    klines = pd.read_parquet(klines_path)[["open", "high", "low", "close"]].dropna()
    if klines.index.tz is not None:
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)

    # Hybrid trigger
    short_mask = (v9["p_short_win"] >= 0.65) & (v9["ldc_signal"] == -1)
    long_mask = (v9["p_long_win"] >= 0.65) & (v9["ldc_signal"] == +1)
    triggers = v9[short_mask | long_mask].copy()
    triggers["direction"] = np.where(short_mask[short_mask | long_mask], "SHORT", "LONG")

    # Compute entry/exit via paper_trading_tpsl
    from research.paper_trading_tpsl import _find_exit_for_signal
    rows = []
    klines_idx = klines.index
    for ts, r in triggers.iterrows():
        if ts not in klines_idx:
            continue
        entry = float(klines.loc[ts, "close"])
        future = klines.loc[klines_idx > ts]
        if len(future) < 1:
            continue
        dir_for_exit = "UP" if r["direction"] == "LONG" else "DOWN"
        ep, bh, reason = _find_exit_for_signal(
            entry_price=entry, direction=dir_for_exit,
            sl_dist=0.003, rr=0.005 / 0.003,
            bars=future, timeout_bars=8,
        )
        if r["direction"] == "LONG":
            gross = ep / entry - 1.0
        else:
            gross = -(ep / entry - 1.0)
        net = gross - 0.0005
        rows.append({
            "signal_time": ts,
            "direction": r["direction"],
            "p_long_win": float(r.get("p_long_win", 0)),
            "p_short_win": float(r.get("p_short_win", 0)),
            "ldc_signal": int(r["ldc_signal"]),
            "entry_price": entry,
            "exit_price": float(ep),
            "exit_reason": reason,
            "bars_held": int(bh),
            "gross_pct": gross,
            "net_pct_maker": net,
            "win": int(gross > 0),
            "paused_at_signal": 0,
            "tp_dist": 0.005,
            "sl_dist": 0.003,
            "horizon_h": 8,
            "model_version": "walkforward_demo",
        })
    return pd.DataFrame(rows)


def render_hybrid_chart(since_days: int = 30, demo: bool = False) -> str:
    """Render TradingView Lightweight chart of hybrid signals.

    demo=True: use walk-forward OOS data instead of MySQL — useful for
    previewing chart appearance before production signals accumulate.
    """
    if demo:
        sigs = _build_demo_signals_from_walkforward()
        # In demo mode show all available data
        if sigs.empty:
            klines = _fetch_klines_for_chart(since_days=180)
        else:
            sig_start = sigs["signal_time"].min()
            from datetime import datetime as _dt, timezone as _tz
            days_back = max(30, (_dt.now(_tz.utc).replace(tzinfo=None) - sig_start).days + 10)
            klines = _fetch_klines_for_chart(since_days=days_back)
    else:
        sigs = _fetch_hybrid_for_chart(since_days=since_days)
        klines = _fetch_klines_for_chart(since_days=since_days)

    if klines.empty:
        return _empty_chart_html("Klines data not available")

    # ── Build candle + marker arrays ──
    candle_data = []
    markers = []
    trade_lines: list[dict] = []  # for TP/SL price lines

    for dt, row in klines.iterrows():
        utc_ts = int(dt.timestamp())
        ts = utc_ts + 8 * 3600  # display as UTC+8
        candle_data.append({
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })

    n_total = n_long = n_short = n_paused = 0
    n_wins = n_losses = n_open = 0
    cum_net_pct = 0.0

    if not sigs.empty:
        for _, s in sigs.iterrows():
            n_total += 1
            sig_dt = s["signal_time"]
            if pd.isna(sig_dt):
                continue
            if sig_dt.tzinfo is None:
                sig_dt = sig_dt.tz_localize("UTC")
            entry_ts = int(sig_dt.timestamp()) + 8 * 3600
            paused = int(s["paused_at_signal"]) == 1
            direction = s["direction"]
            entry_price = float(s["entry_price"])

            # Entry marker
            if paused:
                n_paused += 1
                color = "#5a5a5a"
                text = "P"
            else:
                if direction == "LONG":
                    n_long += 1
                    color = "#26a69a"
                    text = f"L p={float(s['p_long_win']):.2f}"
                else:
                    n_short += 1
                    color = "#ef5350"
                    text = f"S p={float(s['p_short_win']):.2f}"

            shape = "arrowUp" if direction == "LONG" else "arrowDown"
            position = "belowBar" if direction == "LONG" else "aboveBar"

            markers.append({
                "time": entry_ts,
                "position": position,
                "color": color,
                "shape": shape,
                "text": text,
                "size": 2,
            })

            # Exit marker if settled
            if not pd.isna(s["exit_price"]):
                exit_price = float(s["exit_price"])
                bars_held = int(s["bars_held"]) if pd.notna(s["bars_held"]) else 0
                exit_dt = sig_dt + pd.Timedelta(hours=bars_held)
                exit_ts = int(exit_dt.timestamp()) + 8 * 3600
                reason = str(s["exit_reason"]) if pd.notna(s["exit_reason"]) else ""

                net_pct_disp = float(s["net_pct_maker"]) * 100
                won = (int(s["win"]) == 1) if pd.notna(s["win"]) else False
                if reason == "tp":
                    ex_color, ex_text = "#00c853", f"TP +{net_pct_disp:.2f}%"
                elif reason == "sl":
                    ex_color, ex_text = "#d50000", f"SL {net_pct_disp:.2f}%"
                elif reason == "dynamic":
                    ex_color = "#00c853" if won else "#d50000"
                    ex_text = f"Cross {net_pct_disp:+.2f}%"
                elif reason == "reverse":
                    ex_color = "#00c853" if won else "#d50000"
                    ex_text = f"Reverse {net_pct_disp:+.2f}%"
                else:
                    ex_color = "#9e9e9e"
                    ex_text = f"{reason} {net_pct_disp:+.2f}%"
                if won:
                    n_wins += 1
                else:
                    n_losses += 1

                if not paused:
                    cum_net_pct += float(s["net_pct_maker"]) * 100

                markers.append({
                    "time": exit_ts,
                    "position": "inBar",
                    "color": ex_color,
                    "shape": "circle",
                    "text": ex_text,
                    "size": 1,
                })

                # Only emit TP/SL price lines when this strategy has barriers
                has_barriers = (float(s["tp_dist"]) > 0) and (float(s["sl_dist"]) > 0)
                if has_barriers:
                    trade_lines.append({
                        "start_time": entry_ts,
                        "end_time": exit_ts,
                        "entry": entry_price,
                        "exit": exit_price,
                        "tp": entry_price * (1 + float(s["tp_dist"])) if direction == "LONG"
                              else entry_price * (1 - float(s["tp_dist"])),
                        "sl": entry_price * (1 - float(s["sl_dist"])) if direction == "LONG"
                              else entry_price * (1 + float(s["sl_dist"])),
                        "direction": direction,
                        "paused": paused,
                    })
            else:
                n_open += 1

    # Sort markers chronologically — TradingView requires it
    markers.sort(key=lambda m: m["time"])

    # ── Header / summary ──
    last_dt = klines.index[-1].astimezone(timezone(timedelta(hours=8)))
    last_time_str = last_dt.strftime("%Y-%m-%d %H:%M UTC+8")
    last_price = float(klines["close"].iloc[-1])

    wr_str = f"{n_wins/(n_wins+n_losses)*100:.1f}%" if (n_wins + n_losses) > 0 else "n/a"
    cum_str = f"{cum_net_pct:+.2f}%" if (n_wins + n_losses) > 0 else "n/a"

    candle_json = json.dumps(candle_data)
    markers_json = json.dumps(markers)
    trades_json = json.dumps(trade_lines)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>BTC Hybrid v9+LDC Signals</title>
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
    border-bottom: 1px solid #1c222b; display: flex; gap: 16px;
    flex-wrap: wrap;
  }}
  #stats .pill {{
    padding: 2px 8px; border-radius: 10px; background: #1c222b;
  }}
  #stats .ok  {{ color: #26a69a; }}
  #stats .bad {{ color: #ef5350; }}
  #chart {{ position: absolute; left: 0; right: 0; bottom: 0; top: 60px; }}
  .legend {{
    position: absolute; left: 8px; top: 70px; font-size: 10px;
    color: #b0b8c4; z-index: 10; pointer-events: none;
    background: rgba(15,20,28,0.85); padding: 6px 10px; border-radius: 4px;
  }}
  .legend span {{ display: inline-block; margin-right: 10px; }}
</style>
</head>
<body>
<div id="header">
  <div class="title">Hybrid v9+LDC Signals</div>
  <div class="info">{last_time_str} | ${last_price:,.0f}</div>
</div>
<div id="stats">
  <span class="pill">Total: {n_total}</span>
  <span class="pill ok">LONG: {n_long}</span>
  <span class="pill bad">SHORT: {n_short}</span>
  <span class="pill" style="color:#9e9e9e;">Paused: {n_paused}</span>
  <span class="pill">Open: {n_open}</span>
  <span class="pill">Settled WR: {wr_str}</span>
  <span class="pill {'ok' if cum_net_pct > 0 else 'bad'}">Cum net: {cum_str}</span>
  <span class="pill">Window: {since_days}d</span>
</div>
<div class="legend">
  <span style="color:#26a69a;">▲ LONG entry</span>
  <span style="color:#ef5350;">▼ SHORT entry</span>
  <span style="color:#00c853;">● TP</span>
  <span style="color:#d50000;">● SL</span>
  <span style="color:#9e9e9e;">● Timeout / Paused</span>
</div>
<div id="chart"></div>

<script>
const candleData = {candle_json};
const markersData = {markers_json};
const tradeData = {trades_json};

const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
  layout: {{ background: {{ color: '#0d1117' }}, textColor: '#b0b8c4' }},
  grid: {{
    vertLines: {{ color: '#1c222b' }},
    horzLines: {{ color: '#1c222b' }},
  }},
  rightPriceScale: {{ borderColor: '#1c222b' }},
  timeScale: {{ borderColor: '#1c222b', timeVisible: true, secondsVisible: false }},
  crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
}});

const candleSeries = chart.addCandlestickSeries({{
  upColor: '#26a69a', downColor: '#ef5350',
  borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350',
}});
candleSeries.setData(candleData);
candleSeries.setMarkers(markersData);

// TP/SL price lines for each settled trade (light, brief)
tradeData.forEach(t => {{
  if (t.paused) return;
  const tpLine = candleSeries.createPriceLine({{
    price: t.tp, color: 'rgba(0,200,83,0.3)', lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: false,
    title: '',
  }});
  const slLine = candleSeries.createPriceLine({{
    price: t.sl, color: 'rgba(213,0,0,0.3)', lineWidth: 1,
    lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: false,
    title: '',
  }});
  // Remove after 30s so chart stays clean — TradingView lightweight doesn't
  // support time-bounded lines, so we just leave them.  In practice with
  // <30 trades over 30d the lines are not too noisy.
}});

chart.timeScale().fitContent();
window.addEventListener('resize', () => chart.applyOptions({{
  width: window.innerWidth, height: window.innerHeight - 60,
}}));
</script>
</body>
</html>"""
    return html


def _empty_chart_html(reason: str) -> str:
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Hybrid Chart — No Data</title>
<style>
  body {{ background: #0d1117; color: #b0b8c4; font-family: -apple-system, sans-serif;
          display: flex; align-items: center; justify-content: center; height: 100vh;
          margin: 0; }}
  .box {{ text-align: center; padding: 32px; }}
  h2 {{ color: #fff; margin-bottom: 12px; }}
  p {{ color: #7a828e; }}
</style>
</head><body>
<div class="box">
  <h2>Hybrid v9+LDC Signals</h2>
  <p>{reason}</p>
  <p style="margin-top: 20px; font-size: 12px;">
    Hybrid 訊號累積中。第一個訊號出現後本頁就會有資料。
  </p>
</div></body></html>"""
