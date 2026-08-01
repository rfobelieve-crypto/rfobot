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

v2.2 (2026-07-30, operator feedback "48hr 資訊 + 明顯低點沒被標到"):
  - window anchored to NOW: default last 48h of candles (--hours / &hours=N),
    not a fixed span from the freeze date;
  - resting pools capped by DISTANCE TO PRICE (nearest 16 above + 16 below),
    not by origin recency — a far-but-obvious low can no longer be crowded
    out by newer session levels;
  - dim "forming" layer: pivot extremes still inside their 10-bar
    confirmation window and the running day/week extremes — the two
    mechanical reasons an obvious low has no pool yet. Drawn faded so the
    map admits they exist while being honest that the engine cannot trade
    them yet.

Usage:
    python research/sweep_failure/shadow_review.py --symbol BNB [--hours 48]
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
MAX_RESTING = 32            # nearest-to-price resting pools (16 up + 16 down)
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


def forming_levels(bars) -> list[dict]:
    """Levels the OPERATOR's eye sees but the engine cannot trade YET — the
    two mechanical reasons an obvious low/high has no pool line (feedback
    2026-07-30):
      pivot-pending   an extreme inside the last PIVOT bars; it becomes a
                      swing pool only after PIVOT more bars confirm it
      period-running  today's / this ISO-week's high-low so far; they become
                      PDH/PDL / PWH/PWL only when the period closes
    Drawn faded on the map so they stop looking like missed marks."""
    H, L = SC.H, SC.L
    n = len(bars)
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    P = SC.PIVOT
    out = []
    for i in range(max(P, n - P), n):
        oth_h = [h[k] for k in range(i - P, n) if k != i]
        oth_l = [lo[k] for k in range(i - P, n) if k != i]
        if oth_h and h[i] >= max(oth_h):
            out.append({"origin": i, "lvl": h[i], "side": 1})
        if oth_l and lo[i] <= min(oth_l):
            out.append({"origin": i, "lvl": lo[i], "side": -1})
    dts = [datetime.fromtimestamp(b[0], tz=timezone.utc) for b in bars]
    for keyfn in (lambda d: d.date(), lambda d: d.isocalendar()[:2]):
        curk = keyfn(dts[-1])
        idxs = [i for i in range(n) if keyfn(dts[i]) == curk]
        hi_i = max(idxs, key=lambda i: h[i])
        lo_i = min(idxs, key=lambda i: lo[i])
        out.append({"origin": hi_i, "lvl": h[hi_i], "side": 1})
        out.append({"origin": lo_i, "lvl": lo[lo_i], "side": -1})
    seen, ded = set(), []
    for p in out:
        k = (p["side"], round(p["lvl"], 10))
        if k not in seen:
            seen.add(k)
            ded.append(p)
    return ded


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


def story(sym: str, t: dict, fl: dict | None = None) -> str:
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
    if fl and fl.get("flow_reject") not in (None, "", "na"):
        lines.append(f"  流特徵: 收回{'✓' if fl['flow_reject'] == '1' else '✗'}"
                     f" · 攻擊 {fl.get('flow_att_min', '?')} 分"
                     f" · 量能 {fl.get('flow_vshock', '?')}x"
                     + ("  << 變體C" if (t.get('b')
                        and fl['flow_reject'] == '1') else ""))
    if fl and fl.get("drv_q") not in (None, "", "na"):
        seg = f"  衍生品: Q{'✓' if fl['drv_q'] == '1' else '✗'}"
        if fl.get("drv_liqburst") not in (None, "", "na"):
            seg += f" · 清算 {fl['drv_liqburst']}x"
        if fl.get("v7_align") not in (None, "", "na"):
            seg += f" · V7站隊 {float(fl['v7_align']):+.4f}"
        if fl.get("drv_gap_oi") not in (None, "", "na"):
            seg += f" · gapOI {fl['drv_gap_oi']}%"
        lines.append(seg)
    if t["exit_ts"]:
        x = datetime.fromtimestamp(t["exit_ts"], timezone.utc)
        lines.append(f"  出場:   {x:%m-%d %H:%M}  netR {t['net']:+.3f}")
    else:
        lines.append("  狀態:   OPEN")
    return "\n".join(lines)


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{sym}USDT · 流動性地圖</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root{{--bg:#0b0e11;--panel:#12161c;--border:#1e242d;--text:#eaecef;--muted:#848e9c;--up:#0ecb81;--down:#f6465d;--accent:#f0b90b}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:13px/1.5 Inter,-apple-system,'Segoe UI','PingFang TC','Microsoft JhengHei',sans-serif}}
#hdr{{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;padding:10px 16px;border-bottom:1px solid var(--border)}}
#hdr b{{font-size:16px;letter-spacing:.02em}}
.tag{{font-size:11px;color:var(--muted);border:1px solid var(--border);border-radius:4px;padding:2px 8px;white-space:nowrap}}
.check{{margin-left:auto;font-size:11px;color:var(--muted)}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;padding:10px 16px 2px}}
.chip{{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:6px 12px;min-width:96px}}
.chip .l{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}}
.chip .v{{font-size:15px;margin-top:2px;font-variant-numeric:tabular-nums;white-space:nowrap}}
.sub{{padding:4px 16px 8px;font-size:11px;color:var(--muted)}}
.legend{{display:flex;flex-wrap:wrap;gap:4px 16px;padding:7px 16px;font-size:11px;color:var(--muted);border-top:1px solid var(--border);border-bottom:1px solid var(--border)}}
.legend span{{white-space:nowrap}}
.dot{{display:inline-block;width:10px;height:3px;border-radius:1px;margin:0 5px 2px 0;vertical-align:middle}}
#c{{height:62vh}}
.pane-t{{padding:8px 16px 2px;font-size:11px;color:var(--muted);border-top:1px solid var(--border)}}
#eq{{height:15vh}}
.twrap{{overflow-x:auto;border-top:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 12px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}}
th{{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:500;background:var(--panel)}}
tr:hover td{{background:#151a21}}
th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5),th:nth-child(10),td:nth-child(10),th:nth-child(11),td:nth-child(11),th:nth-child(13),td:nth-child(13){{text-align:right}}
.b{{color:var(--up)}} .win{{color:var(--up)}} .loss{{color:var(--down)}} .open{{color:var(--accent)}} .dim{{color:var(--muted)}}
details{{padding:8px 16px 14px;font-size:11px;color:var(--muted);line-height:1.8}}
summary{{cursor:pointer;user-select:none}}
</style></head><body>
<div id="hdr"><b>{sym}USDT</b><span class="tag">永續 1H · Shadow 流動性地圖</span><span class="tag">{hours}h</span><span class="tag">UTC+8</span><span class="check">{check}</span></div>
{perf}
<div class="legend">
<span><span class="dot" style="background:var(--down)"></span>買側流動性(高點上方→掃了做空)</span>
<span><span class="dot" style="background:var(--up)"></span>賣側流動性(低點下方→掃了做多)</span>
<span>┄ 未被掃</span>
<span>─ 已被掃</span>
<span style="color:var(--accent)">⏳ 已掃·等回踩(8根內)</span>
<span>▲▼ 進場·字母=變體(A灰=僅記錄 / B·C=方向色 / <span style="color:var(--accent)">D=金</span> / +E=盤感線)</span>
<span>● 出場(綠賺紅虧·同字母)</span>
<span>淡色┄ 形成中(引擎尚不可交易)</span>
</div>
<div id="c"></div>
<div class="pane-t" id="eqt">累積 netR（凍結後已平倉 · <span class="dim">─ 全部</span> · <span style="color:var(--up)">─ 變體B</span>）</div>
<div id="eq"></div>
<div class="twrap"><table><thead><tr><th>進場(UTC+8)</th><th>池子</th><th>方向</th><th>價位</th><th>穿越ATR</th><th>B</th><th>C</th><th>D</th><th>收回</th><th>攻擊</th><th>量能</th><th>狀態</th><th>netR</th></tr></thead><tbody>{rows}</tbody></table></div>
<details><summary>白話說明（代號、提名組合、數字怎麼讀）</summary>
<b>代號＝圖上看的一件事：</b><br>
R 收回＝刺穿價位後一小時內縮回內側（拿完止損就回來）　V 放量＝攻擊那幾分鐘的量比這個幣自己平常大　Q 止損驅動＝OI下降＋買賣壓順突破方向打（掃止損不是新錢）　快＝攻擊5分鐘內結束　PA＝V7模型跟fade同邊　LIQ＝清算爆量<br>
<b>八個提名組合（歷史入場券，非證據）：</b><br>
R∧V 放量刺縮回(+0.165/n390)　R∧Q 縮回＋確認掃止損(+0.199/n202)　R∧V∧Q 三重確認·最肥(+0.267/WR70%/n112)　R∧快 五分鐘搶完就跑(+0.115/n531)　R∧快∧Q 快閃＋止損確認(+0.202)　R 有縮回就算(+0.085/n926)　PA V7也站這邊(+0.104)　V∧LIQ 放量＋清算噴(+0.168)<br>
<b>數字怎麼讀：</b>n=歷史出現次數　netR=每筆平均賺幾成停損距離(已扣費, +0.165=16.5%)　t=多不像運氣(2有料/5很硬)　WR=勝率<br>
<b>共同核心：</b>八個提名六個含 R——「刺出去、拿完止損、又縮回來」就是獵殺完成的簽名，其他條件都是加分。<br>
<b>上線判準：</b>歷史成績只是入場券；shadow 每小時記 forward 成績單，樣本夠＋CI低緣>0＋十月預註冊蓋章才談上線。
</details>
<details><summary>圖例與定義</summary>
線起點=造出該價位的針尖K棒 · 變體B=掃單穿越≤0.25 ATR（已註冊濾網） · 形成中=樞紐未滿10根確認／當日當週極值未收盤<br>
變體C（07-31 註冊）= B∧收回✓（1m 價格路徑確認）；變體D（08-01 註冊）= C∧量能高 = 研究的訂單流反轉配方（收回∧量能），量能高=vshock≥該幣自身此前獵取的中位數（因果·零參數·≥5 筆先例才判定）；各 cohort 同 gate 算術並列，升格需自己的 forward 證據 · 流特徵（前瞻記錄，不參與 gate）：收回=獵取小時內 1m 收回價位內側 · 攻擊=突破價位的分鐘數 · 量能=攻擊分鐘量／24h 中位分鐘量 · 反轉配方候選=收回✓+量能高（10 月預註冊驗）<br>
成本=情境A（進場7／時間出場3／停損10 bps） · 網址參數：&hours=12-2160 · &live=60 自動更新
</details>
<script>
const chart=LightweightCharts.createChart(document.getElementById('c'),{{layout:{{background:{{color:'#0b0e11'}},textColor:'#848e9c',fontSize:11}},grid:{{vertLines:{{color:'#151a21'}},horzLines:{{color:'#151a21'}}}},rightPriceScale:{{borderColor:'#1e242d'}},timeScale:{{timeVisible:true,secondsVisible:false,rightOffset:3,borderColor:'#1e242d'}},watermark:{{visible:true,text:'{sym}USDT · shadow',color:'rgba(234,236,239,0.045)',fontSize:42}}}});
const cs=chart.addCandlestickSeries({{upColor:'#0ecb81',downColor:'#f6465d',borderVisible:false,wickUpColor:'#0ecb81',wickDownColor:'#f6465d'}});
cs.setData({candles});
cs.setMarkers({markers});
for(const g of {levels}){{
  const ls=chart.addLineSeries({{color:g.c,lineWidth:g.w,lineStyle:g.st,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false}});
  ls.setData(g.pts);
}}
chart.timeScale().fitContent();
const eqa={eq_all}, eqb={eq_b};
if(eqa.length>1){{
  const ec=LightweightCharts.createChart(document.getElementById('eq'),{{layout:{{background:{{color:'#0b0e11'}},textColor:'#848e9c',fontSize:10}},grid:{{vertLines:{{color:'#151a21'}},horzLines:{{color:'#151a21'}}}},rightPriceScale:{{borderColor:'#1e242d'}},timeScale:{{timeVisible:true,secondsVisible:false,borderColor:'#1e242d'}}}});
  ec.addLineSeries({{color:'#848e9c',lineWidth:1,lastValueVisible:true,priceLineVisible:false}}).setData(eqa);
  if(eqb.length>1)ec.addLineSeries({{color:'#0ecb81',lineWidth:2,lastValueVisible:true,priceLineVisible:false}}).setData(eqb);
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
    ap.add_argument("--hours", type=int, default=48,
                    help="candle window: last N hours (12-2160; cache holds "
                         "~900 days if you ever need more)")
    args = ap.parse_args()
    sym = args.symbol.upper()
    hours = max(12, min(2160, args.hours))

    bars, trades, pool_rows = rederive(sym)
    check = crosscheck(sym, trades)
    print(check)
    # flow annotation lives in the shadow CSV (prospective columns written
    # by the hourly recorder) — keyed the same way the crosscheck matches
    flow: dict[tuple[str, int], dict] = {}
    if LOG.exists():
        with LOG.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["symbol"] == sym:
                    flow[(r.get("level_kind", "swing"), int(r["fill_ts"]))] = r
    fwd = [t for t in trades if t["fill_ts"] >= FREEZE_TS]
    print(f"{sym}: {len(fwd)} forward signals since freeze\n")
    for t in fwd[-3:]:
        print(story(sym, t, flow.get((t["kind"], t["fill_ts"]))))
        print()

    # ── performance panel (the point of watching a shadow at all) ────────
    closed = [t for t in fwd if t["net"] is not None]
    bsub = [t for t in fwd if t["b"]]
    bclosed = [t for t in closed if t["b"]]

    def chip(label, val, cls=""):
        return (f"<div class='chip'><div class='l'>{label}</div>"
                f"<div class='v{(' ' + cls) if cls else ''}'>{val}</div></div>")

    def _wr(ts):
        return (100 * sum(1 for t in ts if t["net"] > 0) / len(ts)) if ts else None

    def _sgn(x):
        return "win" if x > 0 else "loss" if x < 0 else ""

    s_all = sum(t["net"] for t in closed)
    s_b = sum(t["net"] for t in bclosed)
    wr_all, wr_b = _wr(closed), _wr(bclosed)
    chips = [
        chip("本幣 平倉/持倉", f"{len(closed)} / {len(fwd) - len(closed)}"),
        chip("本幣 勝率", f"{wr_all:.0f}%" if wr_all is not None else "—"),
        chip("本幣 ΣnetR", f"{s_all:+.2f}", _sgn(s_all)),
        chip("變體B 平倉/持倉", f"{len(bclosed)} / {len(bsub) - len(bclosed)}"),
        chip("變體B 勝率", f"{wr_b:.0f}%" if wr_b is not None else "—"),
        chip("變體B ΣnetR", f"{s_b:+.2f}", _sgn(s_b)),
    ]
    cclosed = [t for t in bclosed
               if (flow.get((t["kind"], t["fill_ts"])) or {}).get(
                   "flow_reject") == "1"]
    s_c = sum(t["net"] for t in cclosed)
    wr_c = _wr(cclosed)
    dclosed = [t for t in cclosed
               if (flow.get((t["kind"], t["fill_ts"])) or {}).get(
                   "flow_vhigh") == "1"]
    s_d = sum(t["net"] for t in dclosed)
    wr_d = _wr(dclosed)
    eclosed = []
    if sym == "BTC":
        try:
            slog2 = SE.read_log()
            epred = SE.variant_e_pred(slog2)
            ekeys = {(r["symbol"], r.get("level_kind", "swing"), int(r["fill_ts"]))
                     for r in slog2.values()
                     if r["symbol"] == "BTC" and r["status"] == "CLOSED"
                     and r["net_r"] != "" and epred(r)}
            eclosed = [t for t in closed
                       if ("BTC", t["kind"], t["fill_ts"]) in ekeys]
        except Exception:
            eclosed = []
    chips += [
        chip("變體C(B∧收回) 平倉", f"{len(cclosed)}"),
        chip("變體C 勝率", f"{wr_c:.0f}%" if wr_c is not None else "—"),
        chip("變體C ΣnetR", f"{s_c:+.2f}", _sgn(s_c)),
        chip("變體D(C∧量能高) 平倉", f"{len(dclosed)}"),
        chip("變體D 勝率", f"{wr_d:.0f}%" if wr_d is not None else "—"),
        chip("變體D ΣnetR", f"{s_d:+.2f}", _sgn(s_d)),
    ]
    if eclosed:
        s_e = sum(t["net"] for t in eclosed)
        chips += [
            chip("變體E(OI↓∧CVD順破∧清算高) 平倉", f"{len(eclosed)}"),
            chip("變體E ΣnetR", f"{s_e:+.2f}", _sgn(s_e)),
        ]
    gate_line = "全籃進度: 見每週一 09:30 PortfolioClocks 報告"
    asof = ""
    try:
        slog = SE.read_log()
        gs = SE.gate_stats(slog)
        asof = max((r.get("first_seen_utc") or "" for r in slog.values()),
                   default="")
        if gs["n_closed"]:
            chips += [
                chip("全籃29幣 B進度", f"{gs['n_closed']} / {gs['floor']}"),
                chip("全籃 均netR", f"{gs['mean_r']:+.3f}", _sgn(gs["mean_r"])),
                chip("CI低緣·日聚類", f"{gs['ci_low']:+.3f}", _sgn(gs["ci_low"])),
                chip("GATE 狀態", "累積中" if gs["status"] == "accumulating"
                     else gs["status"]),
            ]
        gate_line = ("全籃(29幣) " + SE.gate_progress(slog)
                     + (f" · log截至 {asof} UTC" if asof else ""))
    except Exception:
        pass
    kind_bits = []
    for k in ("swing", "session", "pdh_pdl", "pwh_pwl"):
        kc = [t for t in closed if t["kind"] == k]
        kind_bits.append(f"{KIND_ZH[k]} {len(kc)}筆"
                         + (f" Σ{sum(t['net'] for t in kc):+.2f}" if kc else ""))
    perf = ("<div class='chips'>" + "".join(chips) + "</div>"
            + "<div class='sub'><b>變體A</b>=原始版·波段池·無濾網（Gate F 正式軌道）"
            + "　<b>變體B</b>=四種池＋淺穿越≤0.25ATR（預註冊 forward 中，表格 B 欄）"
            + "　<b>變體C</b>=B∧收回內側✓（1m 價格收回確認，表格 C 欄）　<b>變體D</b>=C∧量能高（收回∧量能=訂單流組合配方，量能高=高於該幣自身歷史中位，表格 D 欄）　<b>變體E</b>=BTC·OI↓∧CVD順破∧清算爆量高＝操作者三面板「當下」讀法（事後段=序列配方另計）</div>"
            + "<div class='sub'>凍結後 forward · 情境A成本 · "
            + " · ".join(kind_bits)
            + (f" · log截至 {asof} UTC" if asof else "") + "</div>")
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

    now_ts = bars[-1][0]
    t0 = now_ts - hours * 3600           # window anchored to NOW, not freeze
    candles = [{"time": b[0] + TZ, "open": b[1], "high": b[2],
                "low": b[3], "close": b[4]} for b in bars if b[0] >= t0]
    bar_ts = [b[0] for b in bars]
    last_close = bars[-1][SC.C]

    def clamp(ts):
        return max(ts, t0)

    markers, levels, rows = [], [], []
    RED, GREEN, GREY, AMBER = "#f6465d", "#0ecb81", "#5b6472", "#f0b90b"

    # variant ladder per trade (A<B<C<D from recorded columns; E via the
    # engine's own membership closure — display shows what is KNOWN so far,
    # blank flow columns fall back to the highest provable tier)
    try:
        _epred = SE.variant_e_pred(SE.read_log())
    except Exception:
        _epred = lambda _r: False  # noqa: E731

    def tier_of(t, fl):
        if not t["b"]:
            return "A"
        if fl.get("flow_reject") == "1":
            if fl.get("flow_vhigh") == "1":
                return "D"
            return "C"
        return "B"

    # ── pools: the map layer ─────────────────────────────────────────────
    # resting pools capped by DISTANCE TO PRICE (not recency): the map's job
    # is "targets in front of price", and recency let fast-churning session
    # levels crowd out far-but-obvious extremes (feedback 2026-07-30)
    rest_all = [p for p in pool_rows if p["state"] == "resting"]
    above = sorted([p for p in rest_all if p["lvl"] >= last_close],
                   key=lambda p: p["lvl"] - last_close)[:MAX_RESTING // 2]
    below = sorted([p for p in rest_all if p["lvl"] < last_close],
                   key=lambda p: last_close - p["lvl"])[:MAX_RESTING // 2]
    resting = above + below
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
        if s_ts >= t0:
            markers.append({"time": s_ts + TZ,
                            "position": "aboveBar" if p["side"] == 1 else "belowBar",
                            "shape": "circle", "color": AMBER, "text": "⏳"})

    # forming layer (faded): what the eye sees but the engine cannot trade yet
    drawn = {round(p["lvl"], 10) for p in resting + waiting}
    F_RED, F_GREEN = "#5c3a40", "#2c4f43"
    forming = [p for p in forming_levels(bars)
               if round(p["lvl"], 10) not in drawn]
    forming = sorted(forming, key=lambda p: abs(p["lvl"] - last_close))[:8]
    for p in forming:
        col = F_RED if p["side"] == 1 else F_GREEN
        levels.append({"c": col, "w": 1, "st": 2,
                       "pts": [{"time": clamp(bar_ts[p["origin"]]) + TZ,
                                "value": p["lvl"]},
                               {"time": now_ts + TZ, "value": p["lvl"]}]})

    # ── forward trades: replay layer (chart shows the window; the table
    #    below always lists every forward trade since freeze) ──────────────
    for t in fwd:
        fl_t = flow.get((t["kind"], t["fill_ts"]), {})
        tier = tier_of(t, fl_t)
        is_e = bool(fl_t) and _epred(fl_t)
        tag = tier + ("+E" if is_e else "")
        line_col = (GREEN if t["side"] == "LONG" else RED) if t["b"] else GREY
        mk_col = (GREY if tier == "A" else AMBER if tier == "D" else line_col)
        end = (t["exit_ts"] or now_ts)
        if end >= t0:                      # off-window lines would stretch
            o_ts = clamp(t["origin_ts"])   # fitContent past the 48h view
            levels.append({"c": line_col, "w": 2 if t["b"] else 1, "st": 0,
                           "pts": [{"time": o_ts + TZ, "value": t["lvl"]},
                                   {"time": end + TZ, "value": t["lvl"]}]})
        if t["fill_ts"] >= t0:
            markers.append({"time": t["fill_ts"] + TZ,
                            "position": "belowBar" if t["side"] == "LONG" else "aboveBar",
                            "shape": "arrowUp" if t["side"] == "LONG" else "arrowDown",
                            "color": mk_col,
                            "text": tag + "·" + KIND_ZH[t["kind"]]})
        if t["exit_ts"] and t["exit_ts"] >= t0:
            win = (t["net"] or 0) > 0
            markers.append({"time": t["exit_ts"] + TZ, "position": "inBar",
                            "shape": "circle",
                            "color": "#0ecb81" if win else "#f6465d",
                            "text": tag})
        f8 = datetime.fromtimestamp(t["fill_ts"] + TZ, timezone.utc)
        stat = ("OPEN" if not t["exit_ts"]
                else ("win" if (t["net"] or 0) > 0 else "loss"))
        fl = flow.get((t["kind"], t["fill_ts"]), {})
        rej = fl.get("flow_reject", "")
        rej_cell = ("<td class='b'>✓</td>" if rej == "1"
                    else "<td>—</td>" if rej == "0"
                    else "<td class='dim'>·</td>")
        att = fl.get("flow_att_min", "")
        vsh = fl.get("flow_vshock", "")
        rows.append(
            f"<tr><td>{f8:%m-%d %H:%M}</td><td>{KIND_ZH[t['kind']]}</td>"
            f"<td>{t['side']}</td><td>{t['lvl']:.6g}</td>"
            f"<td>{t['pierce']:.2f}</td>"
            f"<td class='b'>{'✓' if t['b'] else ''}</td>"
            f"<td class='b'>{'✓' if (t['b'] and rej == '1') else ''}</td>"
            f"<td class='b'>{'✓' if (t['b'] and rej == '1' and fl.get('flow_vhigh') == '1') else ''}</td>"
            + rej_cell
            + f"<td>{att + '分' if att else '·'}</td>"
            f"<td>{vsh + 'x' if vsh else '·'}</td>"
            f"<td class='{stat.lower() if stat != 'OPEN' else 'open'}'>{stat}</td>"
            f"<td>{('%+.3f' % t['net']) if t['net'] is not None else '—'}</td></tr>")

    markers.sort(key=lambda m: m["time"])
    out = RESULTS / f"shadow_review_{sym.lower()}.html"
    out.write_text(HTML.format(
        sym=sym, hours=hours, check=check, candles=json.dumps(candles),
        markers=json.dumps(markers), levels=json.dumps(levels),
        perf=perf, eq_all=json.dumps(eq_all), eq_b=json.dumps(eq_b),
        rows="".join(reversed(rows))), encoding="utf-8")
    print(f"chart -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
