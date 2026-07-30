# -*- coding: utf-8 -*-
"""Shadow review v2 — a LIVING liquidity map, not just a trade replay.

Operator feedback (2026-07-30) driving this version:
  1. level lines must START AT THE WICK TIP that created them (v1 started at
     the sweep bar, so lines floated next to candles and looked wrong);
  2. a legend explaining every glyph belongs ON the page;
  3. the whole thing should work like the LMSR dynamic map: RESTING pools
     drawn dotted and extending to the right edge until price takes them,
     raided pools cut off solid at the raid, plus ?live=<sec> auto-refresh
     (same mechanism as the cancel-flow live chart).

Pool states drawn:
  resting        dotted line from its origin wick to the right edge — an
                 untouched target sitting in front of price
  swept_waiting  the actionable moment: price pierced the level within the
                 last W bars and the engine is waiting for the retest —
                 bright line to the edge + an hourglass marker at the sweep
  raided         solid line from origin to the sweep (consumed); if the
                 retest filled, entry/exit markers follow

Colour language (matches the LMSR map the user trades from):
  red   = buy-side liquidity (above highs; raiding it fades SHORT)
  green = sell-side liquidity (below lows; raiding it fades LONG)
  bright = variant-B grade (pierce <= 0.25 ATR), grey = record-only

Verification kept from v1: an independent re-derivation of every forward
trade is cross-checked against the shadow CSV, and the bar-by-bar story
prints in the thesis's own vocabulary.

v2.1 (2026-07-30, operator request "把績效都放在裡面"):
  - costs now scenario A (Gate F spec) instead of LT's flat taker — review
    numbers are identical to the shadow log, and the crosscheck now compares
    net_r too (guards the cost model, the drift class found today);
  - performance panel in the header: this symbol's forward stats (all vs
    variant B), per-pool-kind totals, and the global 29-coin Variant B gate
    progress line from the shadow CSV;
  - cumulative netR equity pane under the price chart (grey=all, green=B).

Usage:
    python research/sweep_failure/shadow_review.py --symbol BNB [--days 6]
Out: research/results/shadow_review_{sym}.html
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
import shadow_engine as SE  # noqa: E402  (scenario-A cost + gate progress)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = Path(__file__).resolve().parents[2] / "research/results"
LOG = RESULTS / "sweep_shadow_log.csv"
FREEZE_TS = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
PIERCE_MAX_B = 0.25
TZ = 8 * 3600
FETCH_DAYS = 900
MAX_RESTING = 30            # newest resting pools kept on the map (clutter cap)
KIND_ZH = {"swing": "波段", "session": "時段", "pdh_pdl": "昨日", "pwh_pwl": "上週"}
SESSIONS = LT.SESSIONS


def ensure_bars(sym: str) -> Path:
    """Fetch/refresh 1H bars from Binance when the cache is absent or stale
    (the Railway image has no cache; the route regenerates per query)."""
    import json as _json
    import time as _time
    import urllib.request
    p = LT.CACHE / f"{sym}USDT_1h.csv"
    now = _time.time()
    if p.exists():
        last = SC.load_csv(str(p))[-1][0]
        if now - last < 2 * 3600:
            return p
        start_ms = (last + 3600) * 1000
    else:
        LT.CACHE.mkdir(parents=True, exist_ok=True)
        start_ms = int((now - FETCH_DAYS * 86400) * 1000)
    rows = {}
    cur = start_ms
    while cur < now * 1000:
        req = urllib.request.Request(
            "https://api.binance.com/api/v3/klines"
            f"?symbol={sym}USDT&interval=1h&startTime={int(cur)}&limit=1000",
            headers={"User-Agent": "shadow-review/2.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = _json.loads(r.read().decode())
        if not d:
            break
        for k in d:
            if int(k[6]) > now * 1000:
                continue
            rows[int(k[0]) // 1000] = (float(k[1]), float(k[2]),
                                       float(k[3]), float(k[4]), float(k[5]))
        cur = int(d[-1][0]) + 3600_000
        if len(d) < 1000:
            break
    if rows:
        mode = "a" if p.exists() else "w"
        with p.open(mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if mode == "w":
                w.writerow(["time", "open", "high", "low", "close", "volume"])
            for ts in sorted(rows):
                w.writerow([ts, *rows[ts]])
    return p


def build_pools_with_origin(bars) -> dict[str, list[dict]]:
    """Every pool with BOTH its activation bar (engine semantics) and its
    ORIGIN bar — the bar whose wick tip defines the price. The line must
    start on that tip (operator feedback #1)."""
    H, L = SC.H, SC.L
    n = len(bars)
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    out: dict[str, list[dict]] = {k: [] for k in
                                  ("swing", "session", "pdh_pdl", "pwh_pwl")}
    P = SC.PIVOT
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            out["swing"].append({"est": i + P + 1, "origin": i, "lvl": h[i], "side": 1})
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            out["swing"].append({"est": i + P + 1, "origin": i, "lvl": lo[i], "side": -1})

    dts = [datetime.fromtimestamp(b[0], tz=timezone.utc) for b in bars]
    for name, (h0, h1) in SESSIONS.items():
        hi = lo_ = None
        hi_i = lo_i = None
        prev_in = False
        for i, dt in enumerate(dts):
            inside = h0 <= dt.hour < h1
            if inside:
                if not prev_in:
                    hi, hi_i, lo_, lo_i = h[i], i, lo[i], i
                else:
                    if h[i] > hi:
                        hi, hi_i = h[i], i
                    if lo[i] < lo_:
                        lo_, lo_i = lo[i], i
            elif prev_in and hi is not None:
                out["session"].append({"est": i, "origin": hi_i, "lvl": hi, "side": 1})
                out["session"].append({"est": i, "origin": lo_i, "lvl": lo_, "side": -1})
                hi = lo_ = None
            prev_in = inside

    for kind, keyfn in (("pdh_pdl", lambda d: d.date()),
                        ("pwh_pwl", lambda d: d.isocalendar()[:2])):
        cur_key = None
        hi = lo_ = None
        hi_i = lo_i = None
        for i, dt in enumerate(dts):
            k = keyfn(dt)
            if cur_key is None:
                cur_key, hi, hi_i, lo_, lo_i = k, h[i], i, lo[i], i
                continue
            if k != cur_key:
                out[kind].append({"est": i, "origin": hi_i, "lvl": hi, "side": 1})
                out[kind].append({"est": i, "origin": lo_i, "lvl": lo_, "side": -1})
                cur_key, hi, hi_i, lo_, lo_i = k, h[i], i, lo[i], i
            else:
                if h[i] > hi:
                    hi, hi_i = h[i], i
                if lo[i] < lo_:
                    lo_, lo_i = lo[i], i
    return out


def rederive(sym: str):
    """(trades, pools) under the frozen rules, each pool carrying its state:
    resting / swept_waiting / raided (+ trade anatomy when the retest filled).
    Trade logic is byte-identical to the scorer; only bookkeeping is added."""
    bars = SC.load_csv(str(ensure_bars(sym)))
    n = len(bars)
    H, L, C = SC.H, SC.L, SC.C
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    a = SC.atr14(bars)
    pools = build_pools_with_origin(bars)
    trades, pool_rows = [], []
    for kind, plist in pools.items():
        pending = sorted(plist, key=lambda p: p["est"])
        last_exit, idx = -1, 0
        live: list[dict] = []
        for j in range(n):
            while idx < len(pending) and pending[idx]["est"] <= j:
                live.append(pending[idx])
                idx += 1
            if a[j] is None or a[j] == 0:
                continue
            hit = [p for p in live
                   if (h[j] > p["lvl"] if p["side"] == 1 else lo[j] < p["lvl"])]
            if not hit:
                continue
            live = [p for p in live if p not in hit]
            for p in hit:
                lvl, s = p["lvl"], p["side"]
                kd, d = s, -s
                fill = None
                for f in range(j + 1, min(j + 1 + SC.W, n)):
                    if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                        fill = f
                        break
                A = a[j]
                pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / A
                waiting = fill is None and j + SC.W >= n - 1
                pool_rows.append({**p, "kind": kind, "sweep": j,
                                  "state": "swept_waiting" if waiting else "raided",
                                  "pierce": pierce})
                if fill is None or fill <= last_exit or fill + 1 >= n:
                    continue
                stop = lvl - d * SC.DIS * A
                R, xb = None, min(fill + SC.HOLD, n - 1)
                for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
                    if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                        R, xb = -1.0, k
                        break
                if R is None:
                    R = d * (cl[xb] - lvl) / (SC.DIS * A)
                last_exit = xb
                done = bars[fill][0] + SC.HOLD * 3600 <= bars[-1][0]
                trades.append({
                    "kind": kind, "origin_ts": bars[p["origin"]][0],
                    "sweep_ts": bars[j][0], "fill_ts": bars[fill][0],
                    "exit_ts": bars[xb][0] if done else None,
                    "side": "SHORT" if d == -1 else "LONG",
                    "lvl": lvl, "atr": A, "stop": stop, "pierce": pierce,
                    # scenario-A costs (Gate F spec), NOT LT's flat taker —
                    # keeps every displayed number identical to the shadow log
                    "net": SE.net_r(R, lvl, A, R <= -1.0) if done else None,
                    "b": pierce <= PIERCE_MAX_B})
        for p in live:                     # never pierced = resting
            pool_rows.append({**p, "kind": kind, "sweep": None,
                              "state": "resting", "pierce": None})
    trades.sort(key=lambda t: t["fill_ts"])
    return bars, trades, pool_rows


def crosscheck(sym: str, trades) -> str:
    import time as _time
    if LOG.exists() and _time.time() - LOG.stat().st_mtime > 3 * 3600:
        return ("shadow log 副本非即時（>3h 舊）— 本圖為同一套凍結規則的"
                "即時重演算；逐筆對帳請在本機跑")
    logged = {}
    if LOG.exists():
        with LOG.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["symbol"] == sym:
                    logged[(r["level_kind"], int(r["fill_ts"]))] = r
    fwd = [t for t in trades if t["fill_ts"] >= FREEZE_TS]
    hit = sum(1 for t in fwd
              if (r := logged.get((t["kind"], t["fill_ts"])))
              and abs(float(r["entry_px"]) - t["lvl"]) < 1e-6)
    miss = len(fwd) - hit
    # net_r alignment guards the COST model (the exact drift class found
    # 2026-07-30: engine scenario-A vs review flat-taker)
    pairs = [(t, logged.get((t["kind"], t["fill_ts"])))
             for t in fwd if t["net"] is not None]
    nm_tot = sum(1 for _t, r in pairs if r and r.get("net_r"))
    nm_ok = sum(1 for t, r in pairs if r and r.get("net_r")
                and abs(float(r["net_r"]) - t["net"]) < 5e-4)
    nm = f", netR 對齊 {nm_ok}/{nm_tot}" if nm_tot else ""
    return (f"對帳: 重演算 {len(fwd)} 筆 forward, {hit} 筆與 shadow log 一致"
            + (f", {miss} 筆不一致 ← 檢查!" if miss else "")
            + nm + ("" if miss or (nm_tot and nm_ok < nm_tot) else " ✓"))


def story(sym: str, t: dict) -> str:
    f = datetime.fromtimestamp(t["fill_ts"], timezone.utc)
    s = datetime.fromtimestamp(t["sweep_ts"], timezone.utc)
    wait = int((t["fill_ts"] - t["sweep_ts"]) / 3600)
    lines = [
        f"{sym} {t['side']}  [{KIND_ZH[t['kind']]}]"
        f"{'  << 變體B' if t['b'] else '  (穿越過深, 僅記錄)'}",
        f"  流動性: {KIND_ZH[t['kind']]}池 {t['lvl']:.6g}",
        f"  獵取:   {s:%m-%d %H:%M} UTC 掃過, 深度 {t['pierce']:.2f} ATR",
        f"  失敗:   {wait} 根內回到價位 -> {f:%m-%d %H:%M} 進場 {t['side']}",
        f"  風控:   停損 {t['stop']:.6g} (3.5xATR), 最多 {SC.HOLD} 根",
    ]
    if t["exit_ts"]:
        x = datetime.fromtimestamp(t["exit_ts"], timezone.utc)
        lines.append(f"  出場:   {x:%m-%d %H:%M}  netR {t['net']:+.3f}")
    else:
        lines.append("  狀態:   OPEN")
    return "\n".join(lines)


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>shadow map {sym}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>body{{margin:0;background:#0e1116;color:#e3e3e3;font-family:'Microsoft JhengHei',monospace}}
#hdr{{padding:8px 14px;font-size:13px;line-height:1.75}} #c{{height:58vh}}
.lg span{{margin-right:14px;white-space:nowrap}}
table{{border-collapse:collapse;font-size:12px;margin:8px 14px}}
td,th{{border:1px solid #2a2f38;padding:3px 8px;text-align:right}}
th{{background:#1a1f27}} td:first-child,th:first-child{{text-align:left}}
.b{{color:#4ade80}} .win{{color:#4ade80}} .loss{{color:#f87171}} .open{{color:#facc15}}
.dim{{color:#8a919c}}</style></head><body>
<div id="hdr"><b>Shadow 流動性地圖 — {sym}</b> (UTC+8 · 凍結後 forward) · {check}<br>
<span class="lg">
<span style="color:#e01b24">━ 紅=買側流動性(高點上方,掃了做空)</span>
<span style="color:#26a269">━ 綠=賣側流動性(低點下方,掃了做多)</span>
<span class="dim">┄ 虛線=尚未被掃(延伸至最右=眼前的目標)</span>
<span>─ 實線=已被掃(截於掃單棒)</span>
<span style="color:#facc15">⏳=已掃,等回踩中(8根內)</span>
</span><br><span class="lg">
<span>▲▼=進場(亮色=變體B 淺穿越≤0.25ATR / 灰=僅記錄)</span>
<span>●=出場(綠賺/紅虧)</span>
<span class="dim">線起點=造出該價位的針尖 K 棒</span>
<span class="dim">網址加 &live=60 自動更新(盯盤)</span>
</span>
<div style="margin-top:6px;padding:6px 10px;background:#141922;border-radius:6px">{perf}</div></div>
<div id="c"></div>
<div id="eqt" class="dim" style="padding:2px 14px;font-size:12px">累積 netR（凍結後已平倉 · 灰=全部 / 綠=變體B）</div>
<div id="eq" style="height:18vh"></div>
<table><tr><th>進場(UTC+8)</th><th>池子</th><th>方向</th><th>價位</th><th>穿越ATR</th><th>B</th><th>狀態</th><th>netR</th></tr>{rows}</table>
<script>
const chart=LightweightCharts.createChart(document.getElementById('c'),{{layout:{{background:{{color:'#0e1116'}},textColor:'#9aa0a6'}},grid:{{vertLines:{{color:'#1c2129'}},horzLines:{{color:'#1c2129'}}}},timeScale:{{timeVisible:true,secondsVisible:false,rightOffset:6}}}});
const cs=chart.addCandlestickSeries({{upColor:'#26a269',downColor:'#e01b24',borderVisible:false,wickUpColor:'#26a269',wickDownColor:'#e01b24'}});
cs.setData({candles});
cs.setMarkers({markers});
for(const g of {levels}){{
  const ls=chart.addLineSeries({{color:g.c,lineWidth:g.w,lineStyle:g.st,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false}});
  ls.setData(g.pts);
}}
chart.timeScale().fitContent();
const eqa={eq_all}, eqb={eq_b};
if(eqa.length>1){{
  const ec=LightweightCharts.createChart(document.getElementById('eq'),{{layout:{{background:{{color:'#0e1116'}},textColor:'#9aa0a6'}},grid:{{vertLines:{{color:'#1c2129'}},horzLines:{{color:'#1c2129'}}}},timeScale:{{timeVisible:true,secondsVisible:false}}}});
  ec.addLineSeries({{color:'#9aa0a6',lineWidth:1,lastValueVisible:true,priceLineVisible:false}}).setData(eqa);
  if(eqb.length>1)ec.addLineSeries({{color:'#4ade80',lineWidth:2,lastValueVisible:true,priceLineVisible:false}}).setData(eqb);
  ec.timeScale().fitContent();
}}else{{document.getElementById('eq').style.display='none';document.getElementById('eqt').style.display='none';}}
</script>
<script>
(function () {{
  var q = new URLSearchParams(location.search);
  var raw = q.get('live');
  if (raw === null || location.protocol === 'file:') return;
  var SEC = Math.max(30, parseInt(raw, 10) || 60);
  var hdr = document.getElementById('hdr');
  var badge = document.createElement('span');
  badge.style.cssText = 'margin-left:8px;padding:1px 8px;border-radius:6px;background:#14321f;color:#4ade80;font-size:12px;';
  badge.textContent = 'LIVE ' + SEC + 's';
  if (hdr) hdr.appendChild(badge);
  var stale = 0;
  function tick() {{
    if (document.hidden) return;
    fetch(location.href, {{cache: 'no-store'}}).then(function (r) {{
      return r.text().then(function (t) {{
        if (r.ok && t.indexOf('id="c"') !== -1) {{
          document.open(); document.write(t); document.close();
        }} else {{ throw new Error('bad render'); }}
      }});
    }}).catch(function () {{
      stale += 1;
      badge.style.background = '#3a2323';
      badge.style.color = '#f87171';
      badge.textContent = 'STALE x' + stale + ' (retry ' + SEC + 's)';
    }});
  }}
  if (window.__liveTimer) clearInterval(window.__liveTimer);
  window.__liveTimer = setInterval(tick, SEC * 1000);
}})();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--days", type=int, default=6)
    args = ap.parse_args()
    sym = args.symbol.upper()

    bars, trades, pool_rows = rederive(sym)
    check = crosscheck(sym, trades)
    print(check)
    fwd = [t for t in trades if t["fill_ts"] >= FREEZE_TS]
    print(f"{sym}: {len(fwd)} forward signals since freeze\n")
    for t in fwd[-3:]:
        print(story(sym, t))
        print()

    # ── performance panel (the point of watching a shadow at all) ────────
    closed = [t for t in fwd if t["net"] is not None]
    bsub = [t for t in fwd if t["b"]]
    bclosed = [t for t in closed if t["b"]]

    def _pstat(ts_all, ts_closed):
        if not ts_closed:
            return f"n={len(ts_all)} (open {len(ts_all)}) · 尚無平倉"
        s = sum(t["net"] for t in ts_closed)
        wr = 100 * sum(1 for t in ts_closed if t["net"] > 0) / len(ts_closed)
        return (f"n={len(ts_all)} (open {len(ts_all) - len(ts_closed)}) · "
                f"closed {len(ts_closed)} · WR {wr:.0f}% · "
                f"ΣnetR {s:+.2f} · 每筆均 {s / len(ts_closed):+.3f}")

    kind_bits = []
    for k in ("swing", "session", "pdh_pdl", "pwh_pwl"):
        kc = [t for t in closed if t["kind"] == k]
        kind_bits.append(f"{KIND_ZH[k]} {len(kc)}筆"
                         + (f" Σ{sum(t['net'] for t in kc):+.2f}" if kc else ""))
    try:
        slog = SE.read_log()
        asof = max((r.get("first_seen_utc") or "" for r in slog.values()),
                   default="")
        gate_line = ("全籃(29幣) " + SE.gate_progress(slog)
                     + (f" · log截至 {asof} UTC" if asof else ""))
    except Exception:
        gate_line = "全籃進度: 見每週一 09:30 PortfolioClocks 報告"
    perf = (f"<b>本幣績效 (凍結後 forward · 情境A成本):</b> {_pstat(fwd, closed)}<br>"
            f"<b>變體B (穿越≤0.25 ATR):</b> {_pstat(bsub, bclosed)}<br>"
            f"<span class='dim'>{' | '.join(kind_bits)}</span><br>"
            f"<span class='dim'>{gate_line}</span>")
    print(gate_line)

    def _dedup(pts):
        d = {}
        for p in pts:
            d[p["time"]] = p["value"]
        return [{"time": k, "value": v} for k, v in sorted(d.items())]

    eq_all, eq_b, acc, accb = [], [], 0.0, 0.0
    for t in sorted(closed, key=lambda x: x["exit_ts"]):
        acc += t["net"]
        eq_all.append({"time": t["exit_ts"] + TZ, "value": round(acc, 4)})
        if t["b"]:
            accb += t["net"]
            eq_b.append({"time": t["exit_ts"] + TZ, "value": round(accb, 4)})
    eq_all, eq_b = _dedup(eq_all), _dedup(eq_b)

    t0 = FREEZE_TS - args.days * 86400
    candles = [{"time": b[0] + TZ, "open": b[1], "high": b[2],
                "low": b[3], "close": b[4]} for b in bars if b[0] >= t0]
    now_ts = bars[-1][0]
    ts_of = {b[0]: b[0] for b in bars}
    bar_ts = [b[0] for b in bars]

    def clamp(ts):
        return max(ts, t0)

    markers, levels, rows = [], [], []
    RED, GREEN, GREY, AMBER = "#e01b24", "#26a269", "#5b6472", "#facc15"

    # ── pools: the map layer ─────────────────────────────────────────────
    resting = sorted([p for p in pool_rows if p["state"] == "resting"],
                     key=lambda p: -p["origin"])[:MAX_RESTING]
    waiting = [p for p in pool_rows if p["state"] == "swept_waiting"]
    for p in resting:
        o_ts = bar_ts[p["origin"]]
        if o_ts < t0:
            o_ts = t0
        col = RED if p["side"] == 1 else GREEN
        levels.append({"c": col, "w": 1, "st": 2,      # dotted, to the edge
                       "pts": [{"time": clamp(o_ts) + TZ, "value": p["lvl"]},
                               {"time": now_ts + TZ, "value": p["lvl"]}]})
    for p in waiting:
        o_ts = clamp(bar_ts[p["origin"]])
        s_ts = bar_ts[p["sweep"]]
        col = RED if p["side"] == 1 else GREEN
        levels.append({"c": col, "w": 2, "st": 0,
                       "pts": [{"time": o_ts + TZ, "value": p["lvl"]},
                               {"time": now_ts + TZ, "value": p["lvl"]}]})
        markers.append({"time": s_ts + TZ, "position": "aboveBar" if p["side"] == 1 else "belowBar",
                        "shape": "circle", "color": AMBER, "text": "⏳"})

    # ── forward trades: replay layer ─────────────────────────────────────
    for t in fwd:
        col = (GREEN if t["side"] == "LONG" else RED) if t["b"] else GREY
        o_ts = clamp(t["origin_ts"])
        end = (t["exit_ts"] or now_ts)
        levels.append({"c": col, "w": 2 if t["b"] else 1, "st": 0,
                       "pts": [{"time": o_ts + TZ, "value": t["lvl"]},
                               {"time": end + TZ, "value": t["lvl"]}]})
        markers.append({"time": t["fill_ts"] + TZ,
                        "position": "belowBar" if t["side"] == "LONG" else "aboveBar",
                        "shape": "arrowUp" if t["side"] == "LONG" else "arrowDown",
                        "color": col,
                        "text": ("B " if t["b"] else "") + KIND_ZH[t["kind"]]})
        if t["exit_ts"]:
            win = (t["net"] or 0) > 0
            markers.append({"time": t["exit_ts"] + TZ, "position": "inBar",
                            "shape": "circle",
                            "color": "#4ade80" if win else "#f87171", "text": ""})
        f8 = datetime.fromtimestamp(t["fill_ts"] + TZ, timezone.utc)
        stat = ("OPEN" if not t["exit_ts"]
                else ("win" if (t["net"] or 0) > 0 else "loss"))
        rows.append(
            f"<tr><td>{f8:%m-%d %H:%M}</td><td>{KIND_ZH[t['kind']]}</td>"
            f"<td>{t['side']}</td><td>{t['lvl']:.6g}</td>"
            f"<td>{t['pierce']:.2f}</td>"
            f"<td class='b'>{'✓' if t['b'] else ''}</td>"
            f"<td class='{stat.lower() if stat != 'OPEN' else 'open'}'>{stat}</td>"
            f"<td>{('%+.3f' % t['net']) if t['net'] is not None else '—'}</td></tr>")

    markers.sort(key=lambda m: m["time"])
    out = RESULTS / f"shadow_review_{sym.lower()}.html"
    out.write_text(HTML.format(
        sym=sym, check=check, candles=json.dumps(candles),
        markers=json.dumps(markers), levels=json.dumps(levels),
        perf=perf, eq_all=json.dumps(eq_all), eq_b=json.dumps(eq_b),
        rows="".join(reversed(rows))), encoding="utf-8")
    print(f"chart -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
