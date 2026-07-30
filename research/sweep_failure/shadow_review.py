# -*- coding: utf-8 -*-
"""Shadow review — SEE what the recorder is recording, and verify it.

The operator's two requirements (2026-07-30):
  1. visibility — the shadow engine is a CSV plus a scheduled task, invisible
     in operation. This renders its signals for one symbol onto 1H candles
     (level line, sweep, entry, exit) as a standalone HTML chart.
  2. verification — "make sure it trades the strategy I have in mind". Two
     mechanisms: (a) a bar-by-bar STORY for recent signals in the exact
     vocabulary of the thesis (哪個池子 -> 掃多深 -> 幾根內回來 -> 進場 ->
     出場); (b) an independent re-derivation of every forward signal that is
     CROSS-CHECKED against the shadow CSV — if the log ever disagrees with
     the frozen rules, this prints the mismatch loudly.

Read-only: reads the kline cache + the shadow CSV, writes one HTML.

Usage:
    python research/sweep_failure/shadow_review.py --symbol BNB
    python research/sweep_failure/shadow_review.py --symbol BTC --days 10
Out: research/results/shadow_review_{SYM}.html  (+ stories on stdout)
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RESULTS = Path(__file__).resolve().parents[2] / "research/results"
FETCH_DAYS = 900          # enough history that swing levels match the local run


def ensure_bars(sym: str) -> Path:
    """The Railway image has no kline cache (gitignored), and a local cache
    can be stale between hourly shadow runs. Fetch/refresh from Binance
    public REST on demand — the route regenerates per query by design."""
    import csv as _csv
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
            headers={"User-Agent": "shadow-review/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = _json.loads(r.read().decode())
        if not d:
            break
        for k in d:
            if int(k[6]) > now * 1000:      # live bar
                continue
            rows[int(k[0]) // 1000] = (float(k[1]), float(k[2]),
                                       float(k[3]), float(k[4]), float(k[5]))
        cur = int(d[-1][0]) + 3600_000
        if len(d) < 1000:
            break
    if rows:
        mode = "a" if p.exists() else "w"
        with p.open(mode, newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if mode == "w":
                w.writerow(["time", "open", "high", "low", "close", "volume"])
            for ts in sorted(rows):
                w.writerow([ts, *rows[ts]])
    return p
LOG = RESULTS / "sweep_shadow_log.csv"
FREEZE_TS = int(datetime(2026, 7, 28, tzinfo=timezone.utc).timestamp())
PIERCE_MAX_B = 0.25
TZ = 8 * 3600                      # display UTC+8, same as the other charts
KIND_ZH = {"swing": "波段高低點", "session": "時段極值",
           "pdh_pdl": "昨日高低", "pwh_pwl": "上週高低"}


def rederive(sym: str) -> list[dict]:
    """Independent re-derivation of every trade with full anatomy
    (sweep bar, side, level, fill, exit) — the same frozen rules the shadow
    uses, but carrying the fields the log does not store."""
    p = ensure_bars(sym)
    bars = SC.load_csv(str(p))
    n = len(bars)
    H, L, C = SC.H, SC.L, SC.C
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    a = SC.atr14(bars)
    out = []

    def run_pool(kind: str, pools: list[tuple[int, float, int]]):
        pending = sorted(pools)
        last_exit, idx = -1, 0
        live: list[tuple[float, int]] = []
        for j in range(n):
            while idx < len(pending) and pending[idx][0] <= j:
                live.append((pending[idx][1], pending[idx][2]))
                idx += 1
            if a[j] is None or a[j] == 0:
                continue
            hit = [(pr, s) for pr, s in live
                   if (h[j] > pr if s == 1 else lo[j] < pr)]
            if not hit:
                continue
            live[:] = [x for x in live if x not in hit]
            for lvl, s in hit:
                kd, d = s, -s
                fill = None
                for f in range(j + 1, min(j + 1 + SC.W, n)):
                    if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                        fill = f
                        break
                if fill is None or fill <= last_exit or fill + 1 >= n:
                    continue
                A = a[j]
                stop = lvl - d * SC.DIS * A
                R, xb = None, min(fill + SC.HOLD, n - 1)
                for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
                    if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                        R, xb = -1.0, k
                        break
                if R is None:
                    R = d * (cl[xb] - lvl) / risk_unit(A)
                last_exit = xb
                pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / A
                done = bars[fill][0] + SC.HOLD * 3600 <= bars[-1][0]
                out.append({
                    "kind": kind, "sweep_ts": bars[j][0],
                    "fill_ts": bars[fill][0],
                    "exit_ts": bars[xb][0] if done else None,
                    "side": "SHORT" if d == -1 else "LONG",
                    "lvl": lvl, "atr": A, "stop": stop, "pierce": pierce,
                    "net": LT.net(R, lvl, A) if done else None,
                    "b": pierce <= PIERCE_MAX_B})

    def risk_unit(A):
        return SC.DIS * A

    # swing pools come from the frozen pivot detector; their "established"
    # bar is the sweep scan start (pivot idx + PIVOT + 1) — but the engine
    # semantics are simply "first pierce after confirmation", which run_pool
    # reproduces when given (confirm_bar, level, side).
    piv = []
    P = SC.PIVOT
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            piv.append((i + P + 1, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            piv.append((i + P + 1, lo[i], -1))
    run_pool("swing", piv)
    lv = LT.build_levels(bars)
    for kind in ("session", "pdh_pdl", "pwh_pwl"):
        run_pool(kind, lv.get(kind, []))
    return sorted(out, key=lambda t: t["fill_ts"])


def crosscheck(sym: str, trades: list[dict]) -> str:
    """The verification: the CSV the scheduler wrote vs this re-derivation.
    On Railway the committed log copy is stale (it is written locally every
    hour, pushed occasionally) — comparing fresh re-derivation against a
    stale log would scream false mismatches, so staleness downgrades the
    check to an honest note instead."""
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
    hit = miss = 0
    for t in fwd:
        row = logged.get((t["kind"], t["fill_ts"]))
        if row and abs(float(row["entry_px"]) - t["lvl"]) < 1e-6:
            hit += 1
        else:
            miss += 1
            print(f"  MISMATCH: {t['kind']} fill={t['fill_ts']} "
                  f"lvl={t['lvl']} not in log or price differs")
    extra = len(logged) - hit
    return (f"log cross-check: {hit}/{len(fwd)} re-derived forward trades "
            f"found in log{'' if not miss else f', {miss} MISSING'}"
            f"{'' if extra <= 0 else f', {extra} log rows not re-derived (stale bars?)'}")


def story(sym: str, t: dict) -> str:
    """One signal in the thesis's own vocabulary."""
    f = datetime.fromtimestamp(t["fill_ts"], timezone.utc)
    s = datetime.fromtimestamp(t["sweep_ts"], timezone.utc)
    wait = int((t["fill_ts"] - t["sweep_ts"]) / 3600)
    lines = [
        f"{sym} {t['side']}  [{KIND_ZH[t['kind']]}]"
        f"{'  << 變體B' if t['b'] else '  (穿越過深, 只記錄不入變體B)'}",
        f"  流動性: {t['kind']} 價位 {t['lvl']:.6g}",
        f"  獵取:   {s:%m-%d %H:%M} UTC 掃過價位, 深度 {t['pierce']:.2f} ATR"
        f" ({'淺=獵殺停損形' if t['b'] else '深=較像真突破'})",
        f"  失敗:   {wait} 根內價格回到價位 -> {f:%m-%d %H:%M} 於 {t['lvl']:.6g} 進場 {t['side']}",
        f"  風控:   停損 {t['stop']:.6g} (3.5xATR), 最多持 {SC.HOLD} 根",
    ]
    if t["exit_ts"]:
        x = datetime.fromtimestamp(t["exit_ts"], timezone.utc)
        lines.append(f"  出場:   {x:%m-%d %H:%M}  netR {t['net']:+.3f}")
    else:
        lines.append(f"  狀態:   OPEN — 結果尚未揭曉")
    return "\n".join(lines)


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>shadow review {sym}</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>body{{margin:0;background:#0e1116;color:#e3e3e3;font-family:'Microsoft JhengHei',monospace}}
#hdr{{padding:8px 14px;font-size:13px;line-height:1.6}} #c{{height:62vh}}
table{{border-collapse:collapse;font-size:12px;margin:8px 14px}}
td,th{{border:1px solid #2a2f38;padding:3px 8px;text-align:right}}
th{{background:#1a1f27}} td:first-child,th:first-child{{text-align:left}}
.b{{color:#4ade80}} .win{{color:#4ade80}} .loss{{color:#f87171}} .open{{color:#facc15}}</style>
</head><body>
<div id="hdr"><b>Shadow 覆盤 — {sym}</b> (凍結後 forward 訊號, UTC+8)<br>
▲▼=進場(綠多/紅空, 亮=變體B 淡=僅記錄) · ●=出場(綠賺/紅虧) · 橫線=被獵取的流動性價位 (由掃單棒延伸到出場)<br>
{check}</div>
<div id="c"></div>
<table><tr><th>進場(UTC+8)</th><th>池子</th><th>方向</th><th>價位</th><th>穿越ATR</th><th>變體B</th><th>狀態</th><th>netR</th></tr>{rows}</table>
<script>
const chart=LightweightCharts.createChart(document.getElementById('c'),{{layout:{{background:{{color:'#0e1116'}},textColor:'#9aa0a6'}},grid:{{vertLines:{{color:'#1c2129'}},horzLines:{{color:'#1c2129'}}}},timeScale:{{timeVisible:true,secondsVisible:false}}}});
const cs=chart.addCandlestickSeries({{upColor:'#26a269',downColor:'#e01b24',borderVisible:false,wickUpColor:'#26a269',wickDownColor:'#e01b24'}});
cs.setData({candles});
cs.setMarkers({markers});
for(const seg of {levels}){{
  const ls=chart.addLineSeries({{color:seg.c,lineWidth:1,lineStyle:seg.st,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false}});
  ls.setData(seg.pts);
}}
chart.timeScale().fitContent();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--days", type=int, default=6,
                    help="chart lookback before the first forward signal")
    args = ap.parse_args()
    sym = args.symbol.upper()

    trades = rederive(sym)
    check = crosscheck(sym, trades)
    print(check)
    fwd = [t for t in trades if t["fill_ts"] >= FREEZE_TS]
    print(f"{sym}: {len(fwd)} forward signals since freeze\n")
    for t in fwd[-3:]:
        print(story(sym, t))
        print()

    # candles for the window
    bars = SC.load_csv(str(ensure_bars(sym)))
    t0 = FREEZE_TS - args.days * 86400
    candles = [{"time": b[0] + TZ, "open": b[1], "high": b[2],
                "low": b[3], "close": b[4]} for b in bars if b[0] >= t0]
    now_ts = bars[-1][0]

    markers, levels, rows = [], [], []
    for t in fwd:
        col_in = "#26a269" if t["side"] == "LONG" else "#e01b24"
        if not t["b"]:
            col_in = "#5b6472"
        markers.append({"time": t["fill_ts"] + TZ,
                        "position": "belowBar" if t["side"] == "LONG" else "aboveBar",
                        "shape": "arrowUp" if t["side"] == "LONG" else "arrowDown",
                        "color": col_in, "text": ("B " if t["b"] else "") + t["kind"][:2]})
        if t["exit_ts"]:
            win = (t["net"] or 0) > 0
            markers.append({"time": t["exit_ts"] + TZ, "position": "inBar",
                            "shape": "circle",
                            "color": "#4ade80" if win else "#f87171", "text": ""})
        end = (t["exit_ts"] or now_ts) + TZ
        levels.append({"c": col_in, "st": 0 if t["b"] else 2,
                       "pts": [{"time": t["sweep_ts"] + TZ, "value": t["lvl"]},
                               {"time": end, "value": t["lvl"]}]})
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
        rows="".join(reversed(rows))), encoding="utf-8")
    print(f"chart -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
