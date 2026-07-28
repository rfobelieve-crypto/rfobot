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
from market_data.tasks.cancel_playbook_watcher import (
    classify_state, compute_features, load_frame, state_color)

SMOOTH_MIN = 15
MAX_LEVEL_LINES = 12
CALM_HEX = "#232936"    # 平靜底格色 — 帶子連續可見但不搶戲
TZ_OFFSET_S = 8 * 3600  # display as UTC+8, same convention as chart_interactive
# Per-symbol output: the Flask route renders on demand, so two concurrent
# requests for different coins would otherwise race on one filename and
# serve each other's chart.
def out_path(symbol: str) -> Path:
    stem = "cancel_flow_review" if symbol == "BTC-USD" else            f"cancel_flow_review_{symbol.split('-')[0].lower()}"
    return PROJECT_ROOT / "research" / "results" / f"{stem}.html"


OUT = PROJECT_ROOT / "research" / "results" / "cancel_flow_review.html"


def _q(conn, sql: str, params=None) -> pd.DataFrame:
    """DB → DataFrame without pd.read_sql: its handling of DictCursor rows
    differs across pandas versions (the container's newer pandas turned the
    column ALIAS into row values). dict rows via pd.DataFrame() are stable."""
    with conn.cursor() as cur:
        cur.execute(sql, params or None)
        rows = cur.fetchall() or []
    return pd.DataFrame(rows)


# Instrument for this render. Set once from --symbol in main(); every loader
# reads it rather than taking a parameter, because they are also called from
# the Flask route and from ad-hoc scripts where threading an argument through
# every call site would be more error-prone than one explicit module state.
SYMBOL = "BTC-USD"


def _binance_symbol() -> str:
    """BTC-USD -> BTCUSDT."""
    return SYMBOL.split("-")[0] + "USDT"


def load_depth(hours: int | None) -> pd.DataFrame:
    conn = get_db_conn()
    try:
        dd = _q(conn,
            "SELECT minute_start_ms ms, bid_add_qty, bid_cancel_qty, "
            "ask_add_qty, ask_cancel_qty FROM depth_deltas_1m "
            "WHERE canonical_symbol=%s AND exchange='binance' "
            "ORDER BY minute_start_ms", (SYMBOL,))
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
            "symbol": _binance_symbol(), "interval": "1m",
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
    # v = 總量(base), tbb = taker 買方主動量 → 賣方 = v - tbb,量價三分法用
    for col in ("o", "h", "l", "c", "v", "tbb"):
        df[col] = df[col].astype(float)
    return df[["open_time", "o", "h", "l", "c", "v", "tbb"]]


def load_strong_signals(start_ms: int, end_ms: int) -> list[dict]:
    # tracked_signals is the V7 direction model, which only runs on BTC.
    # Drawing BTC's entries on an ETH chart would be actively misleading.
    if SYMBOL != "BTC-USD":
        return []
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


def load_playbook_events(start_ms: int, end_ms: int) -> list[dict]:
    """Machine-detected playbook events (cancel_playbook_watcher) as markers.

    Mirrors the static monitor's _overlay_playbooks so both charts tell the
    same story (CLAUDE.md 圖表同步規則). These are the WATCHER's own calls —
    distinct from load_strong_signals' "S" markers, which are our v7 entries.
    ONLY alerted=1 rows are drawn (user, 2026-07-27): the chart mirrors what
    actually reached Telegram, so a review can ask "that push I got — what
    did price do next?". Silently-logged rows stay in the DB for statistics
    but off the chart. Deliberately minimal (user: 簡單一點): one dot for
    every playbook, only direction encoded — green under the bar = bullish,
    red over the bar = bearish."""
    try:
        conn = get_db_conn()
        try:
            ev = _q(conn,
                "SELECT minute_start_ms ms, direction "
                "FROM cancel_playbook_events WHERE direction IN ('UP','DOWN') "
                "AND alerted=1 AND canonical_symbol=%s "
                "AND minute_start_ms BETWEEN %s AND %s",
                params=(SYMBOL, start_ms, end_ms))
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"(playbook overlay skipped: {e})")
        return []
    out = []
    for _, r in ev.iterrows():
        ts = int(pd.to_numeric(r["ms"]) // 1000) + TZ_OFFSET_S
        up = r["direction"] == "UP"
        out.append({"time": ts,
                    "position": "belowBar" if up else "aboveBar",
                    "shape": "circle",
                    "color": "#26a269" if up else "#e01b24",
                    "text": ""})
    return sorted(out, key=lambda m: m["time"])


def build_state_strip(start_ms: int, end_ms: int) -> list[dict]:
    """A2-1 (2026-07-18): per-minute display state under each candle.

    classify_state = the frozen v1 classifier's 1:1 relabelling (single
    state function shared with /cancelstate and the watcher) — the ribbon
    adds NO new thresholds. calm minutes draw nothing (blank = 平靜).
    """
    try:
        now_ms = int(time.time() * 1000)
        lookback = int((now_ms - start_ms) / 60_000) + 120
        wf = load_frame(lookback_min=lookback, symbol=SYMBOL)
        if wf.empty:
            return []
        feat = compute_features(wf)
        m0, m1 = start_ms // 60_000, end_ms // 60_000
        out = []
        for m, r in feat.loc[(feat.index >= m0) & (feat.index <= m1)].iterrows():
            # calm 也畫一格極暗底色——帶子連續可見(非平靜佔比 ~1%,全空白
            # 會讓整層看起來不存在, 2026-07-18 UX 修正)
            col = state_color(classify_state(r)) or CALM_HEX
            out.append({"time": int(m) * 60 + TZ_OFFSET_S,
                        "value": 1, "color": col})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"(state strip skipped: {e})")
        return []


def load_hunt_events(start_ms: int, end_ms: int) -> tuple[list[dict], list[dict]]:
    """A2-2 (2026-07-18): liquidity-hunt marker layer.

    Sources: tv_alert_events (user-drawn levels via /tv, state stamped by
    the poller) + liquidity_events (legacy sweep pipeline). Both tables are
    empty until TV alerts are wired — defensive by design.
    Returns (markers ⚡, level price lines)."""
    markers, levels = [], []
    if SYMBOL != "BTC-USD":
        return [], []          # tv_alert_events / liquidity_events are BTC-only
    try:
        conn = get_db_conn()
        try:
            tv = _q(conn,
                "SELECT received_ms ms, event, liquidity_side side, price, state "
                "FROM tv_alert_events WHERE received_ms BETWEEN %s AND %s "
                "AND price IS NOT NULL ORDER BY received_ms",
                params=(start_ms, end_ms))
            try:
                liq = _q(conn,
                    "SELECT trigger_ts ms, liquidity_side side, entry_price price "
                    "FROM liquidity_events WHERE trigger_ts BETWEEN %s AND %s "
                    "ORDER BY trigger_ts", params=(start_ms, end_ms))
            except Exception:
                liq = pd.DataFrame()
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"(hunt overlay skipped: {e})")
        return [], []
    frames = [f for f in (tv, liq) if not f.empty]
    for f in frames:
        for _, r in f.iterrows():
            try:
                ts = int(pd.to_numeric(r["ms"])) // 1000 + TZ_OFFSET_S
                px = float(r["price"])
            except (TypeError, ValueError):
                continue
            side = str(r.get("side") or "").strip()
            label = str(r.get("event") or side or "level")
            state = str(r.get("state") or "")
            markers.append({
                "time": ts,
                "position": "aboveBar" if side == "buy" else "belowBar",
                "shape": "circle", "color": "#f2b544", "size": 1,
                "text": "⚡" + (state if state not in ("", "calm") else ""),
            })
            levels.append({"price": round(px, 1), "title": label[:16]})
    # de-dup level lines by price, keep the most recent few
    seen, uniq = set(), []
    for lv in reversed(levels):
        if lv["price"] in seen:
            continue
        seen.add(lv["price"])
        uniq.append(lv)
    return sorted(markers, key=lambda m: m["time"]), uniq[:MAX_LEVEL_LINES]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=0, help="0 = full depth era")
    ap.add_argument("--symbol", default="BTC-USD",
                    help="canonical symbol, e.g. BTC-USD / ETH-USD. Needs all "
                         "three source tables — depth_deltas_1m, flow_bars_1m "
                         "and orderbook_snapshots_1m — for that instrument.")
    ap.add_argument("--shock-dots", action="store_true",
                    help="overlay amber dots on candles where cancel volume "
                         ">= 3x trailing-60m median (off by default)")
    args = ap.parse_args()
    global SYMBOL
    SYMBOL = args.symbol.strip().upper()

    dd = load_depth(args.hours or None)
    if dd.empty or len(dd) < 30:
        print(f"depth_deltas too young ({len(dd)} min)")
        return 0

    tot = dd["bid_cancel_qty"] + dd["ask_cancel_qty"]
    skew = (dd["ask_cancel_qty"] - dd["bid_cancel_qty"]) / tot.replace(0, np.nan)
    skew = skew - skew.mean()  # remove structural ask-heavy bias, cf. plot_cancel_flow
    skew_s = skew.rolling(SMOOTH_MIN, min_periods=max(3, SMOOTH_MIN // 3)).mean()
    # 淨偏斜 = (賣側淨抽離 − 買側淨抽離)/總撤,淨抽離 = 撤 − 加。
    # 換防(撤多少補多少)時歸零 → 拆穿毛撤單的假象;毛高+淨高才是真撤離。
    # display-only:凍結檢定(cancel_lead_ic / cancel_shock_ic)不用此欄。
    net_num = ((dd["ask_cancel_qty"] - dd["ask_add_qty"])
               - (dd["bid_cancel_qty"] - dd["bid_add_qty"]))
    net_skew = net_num / tot.replace(0, np.nan)
    net_skew = net_skew - net_skew.mean()
    net_s = net_skew.rolling(SMOOTH_MIN, min_periods=max(3, SMOOTH_MIN // 3)).mean()
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

    candles = []
    vol_bars = []
    for t, o, h, lo, c, v, tb in kl.itertuples(index=False):
        ts = int(t) // 1000 + TZ_OFFSET_S
        candles.append({"time": ts, "open": o, "high": h, "low": lo, "close": c})
        # taker 買佔優(2*tbb ≥ v)綠 / 賣佔優紅 — 配撤單面板做「真空/真破/吸收」三分
        vol_bars.append({"time": ts, "value": round(v, 3),
                         "color": "rgba(38,162,105,0.6)" if 2 * tb >= v
                                  else "rgba(224,27,36,0.6)"})

    skew_bars, net_bars, int_bars, shock_bars = [], [], [], []
    for i in range(len(dd)):
        ts = int(dd["ms"].iloc[i]) // 1000 + TZ_OFFSET_S
        z = skew_s.iloc[i]
        if not np.isnan(z):
            deep = abs(z) >= 0.30
            color = (("#26a269" if deep else "#1e5c3f") if z >= 0
                     else ("#e01b24" if deep else "#7a2028"))
            skew_bars.append({"time": ts, "value": round(float(z), 4), "color": color})
        nz = net_s.iloc[i]
        if not np.isnan(nz):
            deep = abs(nz) >= 0.30
            color = (("#26a269" if deep else "#1e5c3f") if nz >= 0
                     else ("#e01b24" if deep else "#7a2028"))
            net_bars.append({"time": ts, "value": round(float(nz), 4), "color": color})
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

    # Marker layering (2026-07-28 使用者定版): the default view carries ONLY
    # the system's own cancel-flow playbook entries — the thing this chart
    # exists to review. Everything else (V7 entry signals, ⚡ liquidity-hunt
    # events, shock dots) is context, not the subject, and four overlapping
    # layers made the read impossible. Context moves behind 專家面板 rather
    # than being deleted, so a review can still pull it up in one click.
    core_markers = load_playbook_events(start_ms, end_ms)
    print(f"playbook (主圖): {len(core_markers)} alerted markers")

    expert_markers = load_strong_signals(start_ms, end_ms)
    if args.shock_dots:
        expert_markers += shock_bars
        print(f"{len(shock_bars)} shock(>=3x) markers -> 專家面板")

    state_strip = build_state_strip(start_ms, end_ms)
    hunt_markers, level_lines = load_hunt_events(start_ms, end_ms)
    expert_markers = sorted(expert_markers + hunt_markers,
                            key=lambda m: m["time"])
    n_coloured = sum(1 for s in state_strip if s["color"] != CALM_HEX)
    print(f"state strip: {len(state_strip)} minutes ({n_coloured} coloured); "
          f"hunt: {len(hunt_markers)} markers -> 專家面板 "
          f"(專家層共 {len(expert_markers)})")
    span_h = (end_ms - start_ms) / 3600_000

    html = HTML_TEMPLATE.format(
        title=f"撤單流覆盤 {SYMBOL} ({span_h:.0f}h)",
        candles=json.dumps(candles), volume=json.dumps(vol_bars),
        skew=json.dumps(skew_bars), netskew=json.dumps(net_bars),
        intensity=json.dumps(int_bars),
        markers_core=json.dumps(core_markers),
        markers_expert=json.dumps(expert_markers),
        # level_lines is intentionally not passed: the price-line layer was
        # dropped 2026-07-19 (too cluttered). load_hunt_events still returns
        # it because the ⚡ markers come from the same query.
        states=json.dumps(state_strip),
        span_h=f"{span_h:.0f}", n=len(dd), smooth=SMOOTH_MIN, sym=SYMBOL)
    out = out_path(SYMBOL)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"saved -> {out}")
    return 0


HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
  body {{ margin:0; background:#0e1116; color:#e3e3e3;
         font-family:'Microsoft JhengHei',sans-serif; }}
  #hdr {{ padding:8px 14px; font-size:14px; color:#c8cdd4; line-height:1.7; }}
  #hdr b {{ color:#ffffff; }}
  #hdr button {{ background:#2a2f38; color:#e3e3e3; border:1px solid #3a4150;
                 border-radius:6px; padding:2px 10px; cursor:pointer;
                 font-size:12px; margin:0 6px; }}
  .muted {{ color:#5f6672; font-size:11px; }}
  .pane {{ position:relative; }}
  .lbl {{ position:absolute; top:6px; left:10px; z-index:5;
          font-size:12px; color:#9aa0a6; pointer-events:none; }}
</style></head><body>
<div id="hdr"><b>撤單流覆盤 {sym}</b> · {span_h}h ·
 <b>看 K 線腳下色格</b>: 🟢讓路看漲 · 🔴抽梯看跌 · 亮灰=有事不判方向 · 深灰=平靜
 · <b>圈點=撤單流劇本進場</b>(綠在K下=看漲 / 紅在K上=看跌，僅推過 TG 的)
 <button id="btnx" onclick="toggleExpert()">🔬 專家面板</button>
 <span class="muted">(專家面板另疊 ⚡獵取事件 · ▲▼V7 訊號)</span>
 <span class="muted">研究非信號 · edge 待 8/10 · n={n}m · {smooth}m 平滑</span></div>
<div class="pane"><div class="lbl">價格 1m K棒 + 狀態色格</div><div id="c1"></div></div>
<div class="pane"><div class="lbl">成交量 (綠=買方主動 / 紅=賣方主動)</div><div id="cv"></div></div>
<div id="expert" style="display:none">
<div class="pane"><div class="lbl">淨偏斜 (綠=賣單牆在變薄 / 紅=買單牆在變薄, 超過虛線才算數)</div><div id="c4"></div></div>
<div class="pane"><div class="lbl">毛撤單偏斜 (＋賣側撤多 / －買側撤多) — 動作;毛高淨零=換防假象</div><div id="c2"></div></div>
<div class="pane"><div class="lbl">撤單強度 (兩側總量)</div><div id="c3"></div></div>
</div>
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
const c1 = mk('c1', 320);
const cv = mk('cv', 80);
const c4 = mk('c4', 100);
const c2 = mk('c2', 100);
const c3 = mk('c3', 80);
let expertOn = false;
function layout() {{
  const H = window.innerHeight - 66;   // header 扣掉後填滿視窗
  const W = window.innerWidth;
  const set = (chart, id, frac, min) => {{
    const h = Math.max(min, Math.floor(H * frac));
    document.getElementById(id).style.height = h + 'px';
    chart.applyOptions({{ width: W, height: h }});
  }};
  if (expertOn) {{
    set(c1, 'c1', 0.42, 260); set(cv, 'cv', 0.10, 60);
    set(c4, 'c4', 0.18, 90);  set(c2, 'c2', 0.16, 80);
    set(c3, 'c3', 0.10, 60);
  }} else {{
    set(c1, 'c1', 0.76, 320); set(cv, 'cv', 0.21, 80);
  }}
}}
function toggleExpert() {{
  expertOn = !expertOn;
  document.getElementById('expert').style.display = expertOn ? 'block' : 'none';
  document.getElementById('btnx').textContent = expertOn ? '收起專家面板' : '🔬 專家面板';
  candle.setMarkers(expertOn ? MK_ALL : MK_CORE);
  layout();
  if (expertOn) {{
    const r = c1.timeScale().getVisibleLogicalRange();
    if (r) [c4, c2, c3].forEach(c => c.timeScale().setVisibleLogicalRange(r));
  }}
}}

const CANDLES = {candles};
const VOL = {volume};
const SKEW = {skew};
const NETSKEW = {netskew};
const INTEN = {intensity};

// 主圖只掛系統自己偵測的撤單流劇本進場點；V7 訊號與 ⚡ 獵取事件屬於背景
// 脈絡，收在專家面板裡，開啟時才疊回來（不是刪掉，覆盤要看隨時叫得出來）。
const MK_CORE = {markers_core};
const MK_EXPERT = {markers_expert};
const MK_ALL = MK_CORE.concat(MK_EXPERT).sort((a, b) => a.time - b.time);

const candle = c1.addCandlestickSeries({{
  upColor:'#26a269', downColor:'#e01b24',
  wickUpColor:'#26a269', wickDownColor:'#e01b24', borderVisible:false }});
candle.setData(CANDLES);
candle.setMarkers(MK_CORE);

// A2-1 狀態色格 — 貼在價格 pane 底部 6% 的 overlay histogram
const STATES = {states};
const stateStrip = c1.addHistogramSeries({{ priceScaleId:'state',
  priceFormat: {{ type:'volume' }}, lastValueVisible:false,
  priceLineVisible:false }});
stateStrip.priceScale().applyOptions({{ scaleMargins: {{ top: 0.90, bottom: 0 }} }});
stateStrip.setData(STATES);

// A2-2 獵取事件只用 ⚡ marker 標示（2026-07-19 使用者定版:
// 價位水平線太亂, 全部移除）

const vol = cv.addHistogramSeries({{ priceFormat: {{ type:'volume' }} }});
vol.setData(VOL);

const skew = c2.addHistogramSeries({{ priceFormat: {{ type:'price', precision:2, minMove:0.01 }} }});
skew.setData(SKEW);
[0.30, -0.30].forEach(v => skew.createPriceLine(
  {{ price: v, color:'#555c66', lineWidth:1, lineStyle:2, title:'' }}));
skew.createPriceLine({{ price: 0, color:'#9aa0a6', lineWidth:1, lineStyle:0, title:'' }});

const netskew = c4.addHistogramSeries({{ priceFormat: {{ type:'price', precision:2, minMove:0.01 }} }});
netskew.setData(NETSKEW);
[0.30, -0.30].forEach(v => netskew.createPriceLine(
  {{ price: v, color:'#555c66', lineWidth:1, lineStyle:2, title:'' }}));
netskew.createPriceLine({{ price: 0, color:'#9aa0a6', lineWidth:1, lineStyle:0, title:'' }});

const inten = c3.addHistogramSeries({{ priceFormat: {{ type:'volume' }} }});
inten.setData(INTEN);

// ── pane sync (visible range + crosshair), guarded against feedback ──
const byTime = arr => {{ const m = new Map();
  arr.forEach(d => m.set(d.time, d.value !== undefined ? d.value : d.close));
  return m; }};
const panes = [
  {{ chart: c1, series: candle,  map: byTime(CANDLES) }},
  {{ chart: cv, series: vol,     map: byTime(VOL) }},
  {{ chart: c2, series: skew,    map: byTime(SKEW) }},
  {{ chart: c4, series: netskew, map: byTime(NETSKEW) }},
  {{ chart: c3, series: inten,   map: byTime(INTEN) }},
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
window.addEventListener('resize', layout);
layout();
c1.timeScale().fitContent();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
