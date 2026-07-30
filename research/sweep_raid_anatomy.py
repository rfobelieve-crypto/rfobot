# -*- coding: utf-8 -*-
"""Raid anatomy — what does the order flow DURING a liquidity raid say about
whether it resolves as REVERSAL or BREAKOUT?

User question (2026-07-30): 每次流動性獵取後，訂單流怎樣最容易反轉、怎樣
容易真突破. This is the flip side of sweep_orderflow.py (which asked "given
a fill, does flow improve the PnL"): here the target is the RESOLUTION of
the raid itself.

Outcome classes (frozen rules, W=8 retest window; the one-position-at-a-time
portfolio constraint is deliberately DROPPED because this characterises the
market phenomenon, not the book):
  BREAKOUT   price never returns to the level within W bars (the sweep held)
  REVERSAL   price retests within W AND the frozen-rules trade from that
             retest ends positive (came back and kept going)
  FAKE_RETEST retests but the trade loses (came back, then bounced away)

Features, all measured in the RAID HOUR ONLY (known at its close, before the
retest can be judged), each signed so + = flow AGREES with the pierce
direction:
  pierce_atr   how far past the level the raid bar went (price feature,
               the incumbent — flow must beat/add to it)
  vshock       raid-hour volume / trailing-24h median minute volume x60
  taker_dir    signed taker delta / volume
  cvd_follow   signed CVD change / volume
  imb_end      signed book imbalance at the raid hour's end
  cancel_skew  signed (ask_cancel-bid_cancel)/total — BTC only, ~3 weeks of
               depth data: UNDERPOWERED, reported as appendix

Mechanical hypotheses stated before looking (the user's own thesis):
  獵殺形 (→ reversal): shallow pierce, volume spike, aggression INTO the
      break that fails to extend price (absorbed) — the stops were the fuel
  真突破形 (→ holds): deep pierce with taker/CVD follow-through and the book
      leaning with the break — real demand, not stop fuel

Discipline: descriptive characterisation, 6 features x 2 cuts ~= expect ~0.6
spurious monotone patterns; only monotone AND BTC/ETH-consistent counts.
Nothing here changes any registration.

Run: python research/sweep_raid_anatomy.py
Out: research/results/sweep_raid_anatomy.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "sweep_failure"))
os.environ["SLIP"] = "0"
import sweep_core as SC  # noqa: E402
import level_types as LT  # noqa: E402
from shared.db import get_db_conn  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT = ROOT / "research/results/sweep_raid_anatomy.json"
SYMS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}


def load_flow(canon: str):
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT window_start ms, volume_usd v, delta_usd d, cvd_usd c "
                "FROM flow_bars_1m WHERE canonical_symbol=%s ORDER BY window_start",
                (canon,))
            fb = cur.fetchall()
            cur.execute(
                "SELECT ts_ms, imbalance_l20 i FROM orderbook_snapshots_1m "
                "WHERE canonical_symbol=%s ORDER BY ts_ms", (canon,))
            ob = cur.fetchall()
            cur.execute(
                "SELECT minute_start_ms ms, bid_cancel_qty b, ask_cancel_qty a "
                "FROM depth_deltas_1m WHERE canonical_symbol=%s "
                "AND exchange='binance' ORDER BY minute_start_ms", (canon,))
            dd = cur.fetchall()
    finally:
        conn.close()
    flow = {int(r["ms"]) // 60_000: (float(r["v"] or 0), float(r["d"] or 0),
                                     float(r["c"] or 0)) for r in fb}
    imb = {int(r["ts_ms"]) // 60_000: float(r["i"])
           for r in ob if r["i"] is not None}
    canc = {int(r["ms"]) // 60_000: (float(r["b"] or 0), float(r["a"] or 0))
            for r in dd}
    return flow, imb, canc


def raids(sym: str) -> list[dict]:
    """Every pool consumption with resolution class, NO position chain."""
    bars = SC.load_csv(str(LT.CACHE / f"{sym}USDT_1h.csv"))
    n = len(bars)
    H, L, C = SC.H, SC.L, SC.C
    h = [b[H] for b in bars]
    lo = [b[L] for b in bars]
    cl = [b[C] for b in bars]
    a = SC.atr14(bars)

    pools: list[tuple[int, float, int]] = []
    P = SC.PIVOT
    for i in range(P, n - P):
        seg = range(i - P, i + P + 1)
        if all(h[i] >= h[k] for k in seg) and any(h[i] > h[k] for k in seg if k != i):
            pools.append((i + P + 1, h[i], 1))
        if all(lo[i] <= lo[k] for k in seg) and any(lo[i] < lo[k] for k in seg if k != i):
            pools.append((i + P + 1, lo[i], -1))
    lv = LT.build_levels(bars)
    for kind in ("session", "pdh_pdl", "pwh_pwl"):
        pools += [(e, p, s) for (e, p, s) in lv.get(kind, [])]

    out = []
    pending = sorted(pools)
    idx = 0
    live: list[tuple[float, int]] = []
    for j in range(n - SC.W - SC.HOLD - 1):
        while idx < len(pending) and pending[idx][0] <= j:
            live.append((pending[idx][1], pending[idx][2]))
            idx += 1
        if a[j] is None or a[j] == 0:
            continue
        hit = [(p, s) for p, s in live if (h[j] > p if s == 1 else lo[j] < p)]
        if not hit:
            continue
        live = [x for x in live if x not in hit]
        for lvl, s in hit:
            kd, d = s, -s
            pierce = (h[j] - lvl if kd == 1 else lvl - lo[j]) / a[j]
            fill = None
            for f in range(j + 1, j + 1 + SC.W):
                if (kd == 1 and lo[f] <= lvl) or (kd == -1 and h[f] >= lvl):
                    fill = f
                    break
            if fill is None:
                # breakout: how far did it extend beyond the level in W bars?
                ext = (max(h[j+1:j+1+SC.W]) - lvl if kd == 1
                       else lvl - min(lo[j+1:j+1+SC.W])) / a[j]
                out.append({"sym": sym, "ts": bars[j][0], "pierce": pierce,
                            "cls": "BREAKOUT", "ext_atr": ext, "netR": None,
                            "side": kd})
                continue
            A = a[j]
            stop = lvl - d * SC.DIS * A
            R, xb = None, min(fill + SC.HOLD, n - 1)
            for k in range(fill + 1, min(fill + SC.HOLD + 1, n)):
                if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                    R, xb = -1.0, k
                    break
            if R is None:
                R = d * (cl[xb] - lvl) / (SC.DIS * A)
            net = LT.net(R, lvl, A)
            out.append({"sym": sym, "ts": bars[j][0], "pierce": pierce,
                        "cls": "REVERSAL" if net > 0 else "FAKE_RETEST",
                        "ext_atr": None, "netR": net, "side": kd})
    return out


def add_flow_feats(rows: list[dict], flow, imb, canc) -> list[dict]:
    out = []
    for r in rows:
        m0 = r["ts"] // 60                      # raid hour = [m0, m0+60)
        win = [flow[m] for m in range(m0, m0 + 60) if m in flow]
        if len(win) < 40:
            continue
        base = [flow[m][0] for m in range(m0 - 1440, m0) if m in flow]
        if len(base) < 500:
            continue
        med = sorted(base)[len(base) // 2]
        vol = sum(w[0] for w in win)
        delta = sum(w[1] for w in win)
        cvd = win[-1][2] - win[0][2]
        s = r["side"]
        f = dict(r)
        f["vshock"] = vol / (med * 60) if med > 0 else 1.0
        f["taker_dir"] = s * (delta / vol) if vol > 0 else 0.0
        f["cvd_follow"] = s * (cvd / vol) if vol > 0 else 0.0
        iv = [imb[m] for m in range(m0 + 50, m0 + 60) if m in imb]
        f["imb_end"] = s * (sum(iv) / len(iv)) if iv else None
        cv = [canc[m] for m in range(m0, m0 + 60) if m in canc]
        if cv:
            b = sum(x[0] for x in cv)
            ask = sum(x[1] for x in cv)
            f["cancel_skew"] = s * ((ask - b) / (ask + b)) if (ask + b) > 0 else None
        else:
            f["cancel_skew"] = None
        out.append(f)
    return out


def terciles(rows, key):
    vals = sorted(r[key] for r in rows if r.get(key) is not None)
    if len(vals) < 90:
        return None
    q1, q2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    g = {"low": [], "mid": [], "high": []}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        g["low" if v <= q1 else ("mid" if v <= q2 else "high")].append(r)
    return g, q1, q2


def profile(rows, key):
    t = terciles(rows, key)
    if not t:
        return None, f"  {key:<12} (insufficient)"
    g, q1, q2 = t
    parts, rec = [], {}
    for k in ("low", "mid", "high"):
        rs = g[k]
        n = len(rs)
        br = 100 * sum(1 for r in rs if r["cls"] == "BREAKOUT") / n
        rv = 100 * sum(1 for r in rs if r["cls"] == "REVERSAL") / n
        nets = [r["netR"] for r in rs if r["netR"] is not None]
        mnet = sum(nets) / len(nets) if nets else float("nan")
        rec[k] = {"n": n, "breakout_pct": br, "reversal_pct": rv,
                  "netR_if_retested": mnet}
        parts.append(f"{k}: 突破{br:.0f}% 反轉{rv:.0f}% netR{mnet:+.3f} (n={n})")
    return rec, f"  {key:<12} 切點{q1:.2f}/{q2:.2f}  " + " | ".join(parts)


def main() -> int:
    print("=" * 78)
    print("  RAID ANATOMY — 掃單當下的訂單流 vs 反轉/突破的解析")
    print("=" * 78)
    allr = []
    for sym, canon in SYMS.items():
        flow, imb, canc = load_flow(canon)
        rr = add_flow_feats(raids(sym), flow, imb, canc)
        print(f"  {sym}: {len(rr)} raids with flow coverage")
        allr += rr
    n = len(allr)
    br = sum(1 for r in allr if r["cls"] == "BREAKOUT")
    rv = sum(1 for r in allr if r["cls"] == "REVERSAL")
    fk = n - br - rv
    print(f"\n  base rates over {n} raids: 突破(無回踩) {100*br/n:.0f}%  "
          f"反轉(回踩且賺) {100*rv/n:.0f}%  假回踩(回踩但虧) {100*fk/n:.0f}%")

    res = {"base": {"n": n, "breakout": br, "reversal": rv, "fake": fk}}
    print(f"\n  [terciles] 每個特徵三桶 — 突破率 / 反轉率 / 回踩後 netR")
    for k in ("pierce", "vshock", "taker_dir", "cvd_follow", "imb_end"):
        rec, line = profile(allr, k)
        res[k] = rec
        print(line)
    print(f"\n  [appendix, BTC-only ~3wk depth data — UNDERPOWERED]")
    rec, line = profile([r for r in allr if r["sym"] == "BTC"], "cancel_skew")
    res["cancel_skew_btc"] = rec
    print(line)

    # split check on the headline features
    print(f"\n  [split] BTC vs ETH（單調性是否兩邊成立）")
    for k in ("pierce", "vshock", "taker_dir"):
        for sym in SYMS:
            _, line = profile([r for r in allr if r["sym"] == sym], k)
            print(f"  {sym:<4}" + line)

    # the user's 2x2: shallow x follow-through
    pv = sorted(r["pierce"] for r in allr)
    q1p = pv[len(pv) // 3]
    tv = sorted(r["taker_dir"] for r in allr)
    q2t = tv[2 * len(tv) // 3]
    print(f"\n  [2x2] 淺穿越(≤{q1p:.2f}ATR) x 追價強(taker_dir前1/3, >{q2t:.2f})")
    for lbl, cond in (
            ("淺+追價弱(獵殺形)", lambda r: r["pierce"] <= q1p and r["taker_dir"] <= q2t),
            ("淺+追價強",        lambda r: r["pierce"] <= q1p and r["taker_dir"] > q2t),
            ("深+追價弱",        lambda r: r["pierce"] > q1p and r["taker_dir"] <= q2t),
            ("深+追價強(真突破形)", lambda r: r["pierce"] > q1p and r["taker_dir"] > q2t)):
        g = [r for r in allr if cond(r)]
        if len(g) < 40:
            print(f"  {lbl:<16} n={len(g)} thin")
            continue
        b = 100 * sum(1 for r in g if r["cls"] == "BREAKOUT") / len(g)
        v = 100 * sum(1 for r in g if r["cls"] == "REVERSAL") / len(g)
        nets = [r["netR"] for r in g if r["netR"] is not None]
        print(f"  {lbl:<16} n={len(g):<5} 突破{b:.0f}%  反轉{v:.0f}%  "
              f"netR|回踩 {sum(nets)/len(nets):+.3f}")

    OUT.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\n  wrote {OUT}")
    print("  READ: 6 特徵 x 多重切法, 預期 ~0.6 個假單調; 只有單調且 BTC/ETH "
          "兩邊一致的才算數。描述性分析, 不改任何註冊。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
